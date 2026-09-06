"""Port of fnlp/MOSS-Speech-Codec `modeling_moss_speech_codec.py`
(snapshot eeec733e): `MossSpeechCodec` (Whisper-VQ encoder + flow/HiFT
decoder) with an inference-only `AudioDecoder`.

Documented deviations from the source (see components/VENDORED_SOURCES.md):
1. `AudioDecoder.__init__` no longer evaluates `flow/config.yaml` through
   `hyperpyyaml`; the exact same module tree is constructed explicitly from
   the frozen parameter set below (transcribed 1:1 from the yaml). This also
   removes the `__set_seed*` load-time side effects of the yaml and the
   training-only `compute_fbank` / `compute_f0` entries.
2. The decoder construction is wrapped in a global-RNG save/restore scope:
   the vendored `CausalConditionalCFM.__init__` still calls
   `set_all_random_seed(0)` and derives its fixed `rand_noise` buffer from
   that stream (kept verbatim for numerical parity), but construction no
   longer pollutes the caller's RNG state.
3. `torchaudio.load` for prompt/reference files is replaced by a soundfile
   backed loader (TorchCodec wheels are unavailable in the target env).
4. Added `decode_from_conditioning(...)`: the tensor-native decode path used
   by the adapter (precomputed prompt codes / mel / xvector); the original
   `decode(prompt_speech=path)` API is kept and delegates to it.
"""

from __future__ import annotations

import logging
import os
import random
import uuid as uuid_module
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import onnxruntime
import soundfile as _sf
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from safetensors.torch import load_file
from torch import nn
from transformers import PreTrainedModel, WhisperFeatureExtractor

from ..cosyvoice_flow.decoder import CausalConditionalDecoder
from ..cosyvoice_flow.flow import CausalMaskedDiffWithXvec
from ..cosyvoice_flow.flow_matching import CausalConditionalCFM
from ..cosyvoice_flow.hifigan_f0_predictor import ConvRNNF0Predictor
from ..cosyvoice_flow.hifigan_generator import HiFTGenerator
from ..cosyvoice_flow.transformer.upsample_encoder import UpsampleConformerEncoder
from ..matcha_components.audio import mel_spectrogram
from .configuration import MossSpeechCodecConfig
from .utils import WhisperVQConfig, extract_speech_token
from .whisper import WhisperVQEncoder

logger = logging.getLogger(__name__)


def _load_audio(path: Union[str, os.PathLike]) -> Tuple[torch.Tensor, int]:
    data, sr = _sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data.copy()).T, sr


@contextmanager
def _scoped_global_rng():
    """Save/restore python/numpy/torch(+cuda) RNG around a side-effecting block."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None and torch.cuda.is_available():
            for dev, state in enumerate(cuda_states):
                torch.cuda.set_rng_state(state, dev)


def _build_flow_decoder(token_frame_rate: float = 12.5, chunk_size: int = 5, token_mel_ratio: int = 4):
    """Explicit construction of the flow/HiFT stack.

    Parameter values are transcribed 1:1 from ``flow/config.yaml`` of the
    locked codec snapshot (eeec733e); only training-only yaml entries
    (compute_fbank/compute_f0/__set_seed*) are omitted. ``cfm_params`` uses a
    simple namespace instead of ``omegaconf.DictConfig`` (attribute access
    only)."""
    cfm_params = SimpleNamespace(
        sigma_min=1e-06,
        solver="euler",
        t_scheduler="cosine",
        training_cfg_rate=0.2,
        inference_cfg_rate=0.7,
        reg_loss_type="l1",
    )
    estimator = CausalConditionalDecoder(
        in_channels=320,
        out_channels=80,
        channels=[256],
        dropout=0.0,
        attention_head_dim=64,
        n_blocks=4,
        num_mid_blocks=12,
        num_heads=8,
        act_fn="gelu",
        static_chunk_size=chunk_size * token_mel_ratio,
        num_decoding_left_chunks=-1,
    )
    decoder = CausalConditionalCFM(
        in_channels=240,
        n_spks=1,
        spk_emb_dim=80,
        cfm_params=cfm_params,
        estimator=estimator,
    )
    encoder = UpsampleConformerEncoder(
        output_size=512,
        attention_heads=8,
        linear_units=2048,
        num_blocks=6,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        attention_dropout_rate=0.1,
        normalize_before=True,
        input_layer="linear",
        pos_enc_layer_type="rel_pos_espnet",
        selfattention_layer_type="rel_selfattn",
        input_size=512,
        upsample_stride=4,
        use_cnn_module=False,
        macaron_style=False,
        static_chunk_size=chunk_size,
    )
    flow = CausalMaskedDiffWithXvec(
        input_size=512,
        output_size=80,
        spk_embed_dim=192,
        output_type="mel",
        vocab_size=20480,
        input_frame_rate=token_frame_rate,
        only_mask_loss=True,
        token_mel_ratio=token_mel_ratio,
        pre_lookahead_len=3,
        encoder=encoder,
        decoder=decoder,
    )
    return flow


def _build_hift():
    return HiFTGenerator(
        in_channels=80,
        base_channels=512,
        nb_harmonics=8,
        sampling_rate=24000,
        nsf_alpha=0.1,
        nsf_sigma=0.003,
        nsf_voiced_threshold=10,
        upsample_rates=[8, 5, 3],
        upsample_kernel_sizes=[16, 11, 7],
        istft_params={"n_fft": 16, "hop_len": 4},
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        source_resblock_kernel_sizes=[7, 7, 11],
        source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        lrelu_slope=0.1,
        audio_limit=0.99,
        f0_predictor=ConvRNNF0Predictor(num_class=1, in_channels=80, cond_channels=512),
    )


def fade_in_out(fade_in_mel, fade_out_mel, window):
    fade_in_mel[..., : fade_in_mel.shape[-1] // 2] = fade_in_mel[
        ..., : fade_in_mel.shape[-1] // 2
    ] * window[: fade_in_mel.shape[-1] // 2] + fade_out_mel[
        ..., -(fade_in_mel.shape[-1] // 2) :
    ] * window[-(fade_in_mel.shape[-1] // 2) :]
    return fade_in_mel


class AudioDecoder(nn.Module):
    """Flow-matching + HiFT vocoder stack with per-uuid streaming caches."""

    def __init__(
        self,
        flow_ckpt_path: Union[str, os.PathLike],
        hift_ckpt_path: Union[str, os.PathLike],
        campplus_model: Union[str, os.PathLike],
        device: Union[str, torch.device] = "cuda",
    ) -> None:
        super().__init__()
        self.device = torch.device(device) if isinstance(device, str) else device

        # Construction is RNG-scoped: the flow constructor seeds global RNG
        # (kept verbatim for rand_noise parity); the caller's state is restored.
        with _scoped_global_rng():
            self.flow = _build_flow_decoder()
            self.flow.load_state_dict(torch.load(flow_ckpt_path, map_location=self.device), strict=False)
            self.hift = _build_hift()
            self.hift.load_state_dict(torch.load(hift_ckpt_path, map_location=self.device))
        self.sample_rate = 24000
        self.feat_extractor = lambda x: mel_spectrogram(
            x,
            n_fft=1920,
            num_mels=80,
            sampling_rate=24000,
            hop_size=480,
            win_size=1920,
            fmin=0,
            fmax=8000,
            center=False,
        )

        self.flow.to(self.device)
        self.hift.to(self.device).eval()
        self.mel_overlap_dict: defaultdict = defaultdict(lambda: None)
        self.hift_cache_dict: defaultdict = defaultdict(lambda: None)
        self.token_min_hop_len = 2 * self.flow.input_frame_rate
        self.token_max_hop_len = 4 * self.flow.input_frame_rate
        self.token_overlap_len = 3.5
        self.mel_overlap_len = int(self.token_overlap_len / self.flow.input_frame_rate * 24000 / (480 * 2))
        self.mel_window = np.hamming(2 * self.mel_overlap_len)
        self.mel_cache_len = 1
        self.source_cache_len = int(self.mel_cache_len * 480)
        session_options = onnxruntime.SessionOptions()
        session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.intra_op_num_threads = 1
        self.campplus_session = onnxruntime.InferenceSession(
            str(campplus_model), sess_opts=session_options, providers=["CPUExecutionProvider"]
        )
        self.speech_window = np.hamming(2 * self.source_cache_len)

    def token2wav(
        self,
        token: torch.Tensor,
        uuid: str,
        prompt_token: Optional[torch.Tensor] = None,
        prompt_feat: Optional[torch.Tensor] = None,
        embedding: Optional[torch.Tensor] = None,
        finalize: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prompt_token = prompt_token if prompt_token is not None else torch.zeros(1, 0, dtype=torch.int32)
        prompt_feat = prompt_feat if prompt_feat is not None else torch.zeros(1, 0, 80)
        embedding = embedding if embedding is not None else torch.zeros(1, 192)

        tts_mel = self.flow.inference(
            token=token.to(self.device),
            token_len=torch.tensor([token.shape[1]], dtype=torch.int32, device=self.device),
            prompt_token=prompt_token.to(self.device),
            prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32, device=self.device),
            prompt_feat=prompt_feat.to(self.device),
            prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32, device=self.device),
            embedding=embedding.to(self.device),
            streaming=False,
            finalize=finalize,
        )

        tts_mel = tts_mel[0]
        if self.mel_overlap_dict[uuid] is not None:
            tts_mel = fade_in_out(tts_mel, self.mel_overlap_dict[uuid], self.mel_window)
        if self.hift_cache_dict[uuid] is not None:
            hift_cache_mel, hift_cache_source = (
                self.hift_cache_dict[uuid]["mel"],
                self.hift_cache_dict[uuid]["source"],
            )
            tts_mel = torch.cat([hift_cache_mel, tts_mel], dim=2)
        else:
            hift_cache_source = torch.zeros(1, 1, 0)

        if not finalize:
            self.mel_overlap_dict[uuid] = tts_mel[:, :, -self.mel_overlap_len :]
            tts_mel = tts_mel[:, :, : -self.mel_overlap_len]
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            self.hift_cache_dict[uuid] = {
                "mel": tts_mel[:, :, -self.mel_cache_len :],
                "source": tts_source[:, :, -self.source_cache_len :],
                "speech": tts_speech[:, -self.source_cache_len :],
            }
            tts_speech = tts_speech[:, : -self.source_cache_len]
        else:
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            del self.hift_cache_dict[uuid]
            del self.mel_overlap_dict[uuid]
        return tts_speech, tts_mel

    def offline_inference(self, token: torch.Tensor) -> torch.Tensor:
        this_uuid = str(uuid_module.uuid1())
        tts_speech, _ = self.token2wav(token, uuid=this_uuid, finalize=True)
        return tts_speech

    def release_session(self, uuid: str) -> None:
        """Idempotently drop per-request streaming caches for `uuid`."""
        self.hift_cache_dict.pop(uuid, None)
        self.mel_overlap_dict.pop(uuid, None)


class MossSpeechCodec(PreTrainedModel):
    """MossSpeech codec model (Whisper-VQ encoder + Flow/HiFT decoder)."""

    config_class = MossSpeechCodecConfig

    def __init__(
        self,
        encoder_weight_path: Union[str, os.PathLike],
        encoder_config_path: Union[str, os.PathLike],
        encoder_feature_extractor_path: Union[str, os.PathLike],
        flow_path: Union[str, os.PathLike],
        device: Union[str, torch.device] = "cuda",
    ) -> None:
        super().__init__(config=MossSpeechCodecConfig())

        # Whisper-VQ encoder
        self.sample_rate = 16000
        config = WhisperVQConfig.from_pretrained(str(encoder_config_path))
        self.whisper_vqmodel = WhisperVQEncoder(config)

        state_dict = load_file(str(encoder_weight_path))
        new_state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("encoder."):
                new_state_dict[k[len("encoder.") :]] = v
        self.whisper_vqmodel.load_state_dict(new_state_dict, strict=False)

        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(str(encoder_feature_extractor_path))

        # Flow / HiFT decoder stack
        self.flow_path = str(flow_path)
        self.audio_decoder = AudioDecoder(
            flow_ckpt_path=os.path.join(self.flow_path, "flow.pt"),
            hift_ckpt_path=os.path.join(self.flow_path, "hift.pt"),
            campplus_model=os.path.join(self.flow_path, "campplus.onnx"),
            device=device,
        ).eval()

    @torch.no_grad()
    def encode(
        self,
        inputs: Union[
            Sequence[Union[str, os.PathLike, Tuple[torch.Tensor, int], torch.Tensor]],
            torch.Tensor,
        ],
        *,
        sampling_rate: Optional[int] = None,
        batch_size: int = 128,
    ) -> List[List[int]]:
        """Encode audio into codec token ids (see hf_codec.utils.extract_speech_token)."""
        if isinstance(inputs, torch.Tensor):
            if inputs.dim() == 2:
                inputs = inputs.unsqueeze(1)
            if inputs.dim() != 3:
                raise ValueError("`inputs` must be (B, C, T) when passing a tensor.")
            sr = sampling_rate or self.sample_rate
            items: List[Tuple[torch.Tensor, int]] = [(inputs[i].squeeze(0).cpu(), sr) for i in range(inputs.size(0))]
        else:
            items = list(inputs)
        return extract_speech_token(self.whisper_vqmodel, self.feature_extractor, items, batch_size=batch_size)

    def _extract_speech_feat(self, speech: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        speech_feat = self.audio_decoder.feat_extractor(speech).squeeze(dim=0).transpose(0, 1)
        speech_feat = speech_feat.unsqueeze(dim=0)
        speech_feat_len = torch.tensor([speech_feat.shape[1]], dtype=torch.int32)
        return speech_feat, speech_feat_len

    def _extract_spk_embedding(self, speech_16k: torch.Tensor) -> torch.Tensor:
        feat = kaldi.fbank(speech_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = self.audio_decoder.campplus_session.run(
            None,
            {self.audio_decoder.campplus_session.get_inputs()[0].name: feat.unsqueeze(0).cpu().numpy()},
        )[0].flatten().tolist()
        return torch.tensor([embedding])

    def compute_voice_conditioning(self, prompt_wav_24k: torch.Tensor) -> dict:
        """Compute the full voice conditioning (codes + mel + xvector).

        `prompt_wav_24k`: (1, T) or (T,) float tensor at 24 kHz. The xvector
        branch resamples to 16 kHz exactly like the reference decode path.
        """
        if prompt_wav_24k.dim() == 1:
            prompt_wav_24k = prompt_wav_24k.unsqueeze(0)
        speech_feat, speech_feat_len = self._extract_speech_feat(prompt_wav_24k)
        speech_token = torch.tensor(self.encode([prompt_wav_24k])[0]).unsqueeze(0)
        token_len = min(int(speech_feat.shape[1] / 4), speech_token.shape[1])
        speech_feat, speech_feat_len[:] = speech_feat[:, : 4 * token_len], 4 * token_len
        speech_token = speech_token[:, :token_len]
        prompt_16k = torchaudio.transforms.Resample(orig_freq=24000, new_freq=16000)(prompt_wav_24k)
        embedding = self._extract_spk_embedding(prompt_16k)
        return {
            "prompt_token": speech_token.int(),
            "prompt_feat": speech_feat,
            "embedding": embedding,
            "prompt_token_len": token_len,
        }

    @torch.no_grad()
    def decode_from_conditioning(
        self,
        audio_codes: Sequence[Sequence[int]],
        conditioning: dict,
        *,
        finalize: bool = True,
    ) -> dict:
        """Tensor-native decode with precomputed voice conditioning."""
        device = self.audio_decoder.device
        prompt_token = conditioning["prompt_token"].to(device)
        prompt_feat = conditioning["prompt_feat"].to(device)
        embedding = conditioning["embedding"].to(device)
        syn_wav_list: List[torch.Tensor] = []
        for codes in audio_codes:
            codes_t = torch.tensor(codes, device=device).unsqueeze(0)
            uuid = os.urandom(16).hex()
            tts_speech, _ = self.audio_decoder.token2wav(
                codes_t,
                uuid=uuid,
                prompt_token=prompt_token,
                prompt_feat=prompt_feat,
                embedding=embedding,
                finalize=finalize,
            )
            syn_wav_list.append(tts_speech.squeeze())
        return {"syn_wav_list": syn_wav_list}

    @torch.no_grad()
    def decode(
        self,
        audio_codes: Union[Sequence[Sequence[int]], torch.LongTensor],
        *,
        prompt_speech: Optional[Union[str, os.PathLike]] = None,
        prompt_speech_sample_rate: Optional[int] = None,
        use_spk_embedding: bool = True,
        use_prompt_speech: bool = True,
        finalize: bool = True,
        device: torch.device = torch.device("cuda"),
    ) -> dict:
        """Reference-compatible file-path decode (delegates to conditioning path)."""
        if isinstance(audio_codes, torch.Tensor):
            if audio_codes.dim() == 3 and audio_codes.size(1) == 1:
                codes_list: List[List[int]] = [audio_codes[i, 0].detach().cpu().tolist() for i in range(audio_codes.size(0))]
            elif audio_codes.dim() == 2:
                codes_list = [row.detach().cpu().tolist() for row in audio_codes]
            else:
                raise ValueError("`audio_codes` must be (B, 1, T) or (B, T) when passing a tensor.")
        else:
            codes_list = [list(c) for c in audio_codes]

        if prompt_speech is None or not os.path.exists(str(prompt_speech)):
            raise ValueError("`prompt_speech` path is required for decoding and must exist.")

        prompt_wav, orig_sr = _load_audio(str(prompt_speech))
        target_sr = self.audio_decoder.sample_rate
        if orig_sr != target_sr:
            prompt_wav = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)(prompt_wav)

        conditioning = self.compute_voice_conditioning(prompt_wav)
        if not use_prompt_speech:
            conditioning["prompt_token"] = torch.zeros(1, 0, dtype=torch.int32)
            conditioning["prompt_feat"] = torch.zeros(1, 0, 80)
        if not use_spk_embedding:
            conditioning["embedding"] = torch.zeros(1, 192)
        return self.decode_from_conditioning(codes_list, conditioning, finalize=finalize)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        *,
        revision: Optional[str] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        use_auth_token: Optional[Union[str, bool]] = None,
        subfolder: Optional[str] = None,
        device: Union[str, torch.device] = "cuda",
        **kwargs,
    ):
        """Instantiate the codec from a local directory (offline discipline).

        Expected layout: ``model.safetensors``, ``config.json``,
        ``preprocessor_config.json``, ``flow/{flow.pt, hift.pt, campplus.onnx}``.
        Hub-repo ids are rejected: materialize weights first (P0 01_version_lock.md).
        """
        path_str = str(pretrained_model_name_or_path)
        if not os.path.isdir(path_str):
            raise FileNotFoundError(
                "MossSpeechCodec.from_pretrained expects a local materialized codec directory "
                f"(got {path_str!r}); download/snapshot handling is intentionally not vendored."
            )
        base = Path(path_str)
        if subfolder:
            base = base / subfolder
        flow_dir = base / "flow"
        missing = []
        for f in ("model.safetensors", "config.json", "preprocessor_config.json"):
            if not (base / f).exists():
                missing.append(str(base / f))
        for f in ("flow.pt", "hift.pt", "campplus.onnx"):
            if not (flow_dir / f).exists():
                missing.append(str(flow_dir / f))
        if missing:
            raise FileNotFoundError("Missing codec assets: " + ", ".join(missing))
        return cls(
            encoder_weight_path=str(base / "model.safetensors"),
            encoder_config_path=str(base / "config.json"),
            encoder_feature_extractor_path=str(base),
            flow_path=str(flow_dir),
            device=device,
        )
