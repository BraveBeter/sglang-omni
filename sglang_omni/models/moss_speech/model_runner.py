# SPDX-License-Identifier: Apache-2.0
"""MOSS-Speech model runner for the OmniScheduler-driven engine (T3.4).

Hook contract (sglang_omni.model_runner.base.ModelRunner):
  custom_prefill_forward: attach the Omni prefill sidecar with dual-channel
      embeddings built from each request's canonical prompt rows (the engine
      builds the REAL ForwardBatch — all scheduler contracts handled).
  post_prefill / post_decode / post_decode_launch: sample BOTH heads per the
      locked reference order (float upcast -> audio constraints ->
      per-channel repetition penalty -> warpers -> per-channel
      argmax/multinomial), apply the audio-mode text overwrite, append the
      row to the request state, stage the feedback embedding, and set
      ``result.next_token_ids`` (which makes the engine skip its own
      sampling — moss_tts precedent).
  before_decode: stage the per-request feedback embeddings into the model's
      ``_decode_input_embedding`` rows and rewrite input_ids to row ids.

Transport (hard-won engine facts, see P3/T3.4 changelog):
  * The engine rebuilds the result object AND the forward_batch between the
    forward and the post hooks — custom fields on either are lost.
  * post_prefill may be REORDERED across other requests' decode steps, and
    duplicate post_prefill deliveries were observed at later generation
    steps. No time-point handoff is safe.
  * The engine's sampler warps next_token_logits in place (sharing storage
    with our text_logits), so slices must be cloned.
  Therefore the model binds per-request dual logits under persistent
  ``(rid, phase)`` keys at forward time, and the post hooks pop exactly
  their entries. The launch path (overlap/async loops) routes through
  post_decode_launch, which runs the same collect and mirrors the base's
  pinned staging buffer for the resolve half.
"""

from __future__ import annotations

import traceback
from typing import Any, List

import torch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.prefill_inputs import (
    OmniPrefillInputs,
    attach_omni_prefill_inputs,
)
from sglang_omni.models.moss_speech import fsm
from sglang_omni.models.moss_speech.request_builders import MossSpeechSGLangRequestData


def _logged(hook: Any) -> Any:
    """Surface full tracebacks in the engine's error channel."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return hook(*args, **kwargs)
        except Exception:
            traceback.print_exc()
            raise

    wrapper.__name__ = getattr(hook, "__name__", "hook")
    return wrapper


def _resolve_rids(requests: Any) -> List[Any]:
    rids = []
    for sr in requests:
        rid = getattr(sr, "rid", None)
        if rid is None:
            rid = getattr(sr, "request_id", None)
        if rid is None:
            rid = getattr(getattr(getattr(sr, "data", None), "req", None), "rid", None)
        if rid is None:
            rid = getattr(getattr(sr, "req", None), "rid", None)
        rids.append(rid)
    return rids


class MossSpeechModelRunner(ModelRunner):
    """Dual-channel (text/audio) generation with the reference FSM."""

    def __init__(self, tp_worker: Any, output_processor: Any) -> None:
        super().__init__(tp_worker, output_processor)
        self._last_transport = "none"

    # ------------------------------------------------------------------ embeds
    def _rows_to_embeds(self, rows: torch.Tensor) -> torch.Tensor:
        """Reference token-driven rule: text != modality_pad -> text embed of
        the text token; else audio embed of the audio-channel id."""
        model = self.model
        rows = rows.to(device=model.embed_tokens.weight.device, dtype=torch.long)
        text_ch, audio_ch = rows[:, 0], rows[:, 1]
        is_audio = text_ch == fsm.MODALITY_PAD
        te = model.embed_tokens(text_ch.masked_fill(is_audio, 0))
        ae = model.audio_embed(audio_ch)
        sel = (~is_audio).unsqueeze(-1).to(te.dtype)
        return te * sel + ae * (1 - sel)

    # ------------------------------------------------------------------ hooks
    @_logged
    def custom_prefill_forward(
        self, forward_batch: Any, schedule_batch: Any, requests: Any
    ) -> None:
        del schedule_batch
        pieces: List[torch.Tensor] = []
        for sched_req in requests:
            data: MossSpeechSGLangRequestData = sched_req.data
            if data.prompt_rows is None:
                raise RuntimeError("MOSS-Speech prefill requires prompt_rows")
            req = data.req
            req_len = int(req.extend_range.length)
            prefix_len = len(req.prefix_indices)
            rows = data.prompt_rows
            if data.output_rows:
                # retracted/re-prefilled request: generated tail participates
                generated = torch.tensor(data.output_rows, dtype=torch.long)
                rows = torch.cat([rows, generated], dim=0)
            current = rows[prefix_len : prefix_len + req_len]
            if int(current.shape[0]) != req_len:
                raise RuntimeError(
                    f"MOSS-Speech prefill row mismatch for {req.rid}: have "
                    f"{int(current.shape[0])}, need {req_len} "
                    f"(prefix={prefix_len}, prompt={int(data.prompt_rows.shape[0])}, "
                    f"generated={len(data.output_rows)})"
                )
            if data.output_rows:
                data.pending_feedback_queue.clear()
            pieces.append(self._rows_to_embeds(current))
        if not pieces:
            embeds = torch.empty(
                (0, self.model.hidden_size),
                device=forward_batch.input_ids.device,
                dtype=self.model.dtype,
            )
        else:
            embeds = torch.cat(pieces, dim=0).to(
                device=forward_batch.input_ids.device, dtype=self.model.dtype
            )
        attach_omni_prefill_inputs(
            forward_batch, OmniPrefillInputs(input_embeds=embeds)
        )
        return None

    @_logged
    def before_decode(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: Any,
        *,
        is_lookahead: bool = False,
    ) -> None:
        del is_lookahead, schedule_batch
        self._write_decode_input_embedding(forward_batch, requests)

    def _write_decode_input_embedding(self, forward_batch: Any, requests: Any) -> None:
        model = self.model
        batch_size = len(requests)
        if batch_size == 0:
            return
        embedding = model._decode_input_embedding
        weight = embedding.weight
        if forward_batch.input_ids.numel() < batch_size:
            raise RuntimeError(
                "MOSS-Speech decode input_ids must contain one row id per request"
            )
        if batch_size > int(weight.shape[0]):
            raise RuntimeError(
                "MOSS-Speech decode batch exceeds staged embedding rows "
                f"({batch_size} > {int(weight.shape[0])})"
            )
        rows = []
        for sched_req in requests:
            queue = sched_req.data.pending_feedback_queue
            if not queue:
                raise RuntimeError(
                    f"Missing decode feedback for {sched_req.data.req.rid}"
                )
            rows.append(queue.popleft() if hasattr(queue, "popleft") else queue.pop(0))
        stacked = torch.stack(rows, dim=0).to(device=weight.device, dtype=weight.dtype)
        with torch.no_grad():
            weight[:batch_size].copy_(stacked)
        row_ids = torch.arange(
            batch_size, dtype=torch.long, device=forward_batch.input_ids.device
        )
        forward_batch.input_ids[:batch_size].copy_(row_ids)

    # ------------------------------------------------------------- step logic
    @_logged
    def post_prefill(
        self, result: Any, forward_batch: Any, schedule_batch: Any, requests: Any
    ) -> None:
        if schedule_batch.is_prefill_only:
            return
        self._collect_step(result, forward_batch, requests, phase="prefill")

    @_logged
    def post_decode(
        self, result: Any, forward_batch: Any, schedule_batch: Any, requests: Any
    ) -> None:
        self._collect_step(result, forward_batch, requests, phase="decode")

    @_logged
    def post_decode_launch(self, result: Any, forward_batch: Any, requests: Any) -> Any:
        """Overlap/async-decode loop entry. The base default would sample the
        text head engine-side and our FSM rows would never advance. Run the
        full dual-channel collect here and mirror the base's pinned staging
        copy for the resolve half."""
        if not requests:
            return None
        self._collect_step(result, forward_batch, requests, phase="decode")
        n = len(requests)
        ids = result.next_token_ids
        if ids is None:
            raise RuntimeError("moss_speech launch produced no next_token_ids")
        host_buf = self._next_host_staging((n,), ids.dtype)
        host_buf[:n].copy_(ids[:n], non_blocking=True)
        return host_buf

    @staticmethod
    def _generator_for(
        data: MossSpeechSGLangRequestData, device: torch.device
    ) -> torch.Generator:
        if data.rng_generator is None:
            gen = torch.Generator(device=device)
            gen.manual_seed(int(data.effective_seed) & ((1 << 63) - 1))
            data.rng_generator = gen
        return data.rng_generator

    def _collect_step(
        self, result: Any, forward_batch: Any, requests: Any, *, phase: str = "prefill"
    ) -> None:
        rids = _resolve_rids(requests)
        # rid+phase persistent store: bound by the forward itself, popped here.
        # Immune to result/fb rebuilds, interleaved forwards of other requests,
        # and engine reordering of post hooks.
        step_store = self.model._dual_logits_by_rid
        per_req = [step_store.pop((r, phase), None) for r in rids]
        if any(p is None for p in per_req):
            missing = [rid for rid, pair in zip(rids, per_req) if pair is None]
            raise RuntimeError(f"Missing MOSS-Speech {phase} logits for {missing}")
        self._last_transport = "rid"
        text_logits = torch.stack([p[0] for p in per_req], dim=0)
        audio_logits = torch.stack([p[1] for p in per_req], dim=0)

        next_tokens: List[int] = []
        for i, sched_req in enumerate(requests):
            data: MossSpeechSGLangRequestData = sched_req.data
            # FSM transition BEFORE sampling reads the last appended row
            last_row = (
                tuple(data.output_rows[-1])
                if data.output_rows
                else tuple(int(v) for v in data.prompt_rows[-1])
            )
            data.mode = fsm.next_mode(last_row, data.mode)

            row = self._select_row(data, text_logits[i], audio_logits[i])
            data.output_rows.append(row)
            selected = fsm.row_selected_token(row)
            next_tokens.append(selected)
            # stage the feedback embedding for the next decode step
            row_t = torch.tensor([row], dtype=torch.long)
            data.pending_feedback_queue.append(self._rows_to_embeds(row_t)[0].detach())
            if fsm.stop_hit(row):
                data.finished = True
                data.stop_reason = "stop_token"
        result.next_token_ids = torch.tensor(
            next_tokens, dtype=torch.long, device=text_logits.device
        )

    def _select_row(self, data: Any, text_logits: Any, audio_logits: Any) -> Any:
        """One real sample; parity drivers may override only in forced mode."""
        t_hist, a_hist = fsm.channel_histories(data.prompt_rows, data.output_rows)
        t_hist = t_hist.to(text_logits.device)
        a_hist = a_hist.to(audio_logits.device)
        gen = (
            self._generator_for(data, text_logits.device)
            if data.params.do_sample
            else None
        )
        text_tok, audio_tok = fsm.sample_row(
            text_logits,
            audio_logits,
            data.params,
            t_hist,
            a_hist,
            len(data.output_rows) + 1,
            data.mode,
            gen,
        )
        return fsm.finalize_row(text_tok, audio_tok, data.mode)

    def lookahead_eligible(self, batch: Any):
        # The private dual-channel history must be committed before next decode.
        return False
