# Standard CLI launch correction (2026-09-09)

The cookbook now launches MOSS-Speech through the installed `sglang-omni serve`
command. `sgl-omni` remains an equivalent supported name. Users supply local AR,
codec and default-voice assets directly, without a developer-specific virtual
environment or a config-generation step. Streaming selects the supplied portable
preset. See sglang-omni/docs/cookbook/moss_speech.md for the user commands.

## Implementation

- sglang-omni/pyproject.toml adds the `sglang-omni` console entry, pointing to the
  same Typer app as `sgl-omni`. Reinstalling an existing source checkout registers
  the alias. It is a package entry point, not a wrapper tied to a local interpreter.
- sglang-omni/sglang_omni/models/moss_speech/config.py exposes model-specific
  `codec_path` and `voice_wav` fields through the existing dynamic CLI parser.
  They become factory arguments only on stages that consume them. Explicit
  top-level options override serialized factory defaults on repeated merges;
  when absent, existing factory arguments and the voice environment fallback work.
- sglang-omni/examples/configs/moss_speech.yaml has no personal paths. The new
  sglang-omni/examples/configs/moss_speech_streaming.yaml selects the existing
  streaming class, retaining its chunk5 routing, memory preset and request limits.
- A real default launch exposed a general discovery gap: AutoConfig could ask
  whether to execute a checkpoint's custom config code. Architecture discovery
  now explicitly sets `trust_remote_code=False` and retains the existing raw
  config fallback. CPU tests prove no input prompt or checkpoint-code execution,
  while built-in config handling remains intact. This shared fix and console
  alias are separate commit ddacdc1; no MOSS-specific behavior enters common CLI.

## Verification

| Check | Result |
|---|---|
| New CLI/config contracts before implementation | 7 expected failures reproduced missing alias/asset wiring and nonportable examples |
| Discovery regression before fix | 2 expected failures reproduced interactive/default trust behavior |
| Final CLI and cross-model config tests | 58 passed, 0 skipped |
| Existing HTTP/config-generator/takeover compatibility tests | 36 passed, 0 skipped |
| Independent CPU-only CI entry, including the 9 new tests | 123 passed, 0 skipped; no SGLang or model assets |
| Installed aliases outside the checkout, without PYTHONPATH | Both `sglang-omni serve --help` and `sgl-omni serve --help` exit0 |
| Final Slurm A800 job3917 | Both configs ready; four modes each, 8 requests total; text/stop equal HF reference, offline WAV equal P5 and streaming PCM equal P6 |
| Noninteractive startup and shutdown | stdin closed; neither profile prompts for remote code; both parent processes exit0 and all 5 observed descendants per launch exit |
| Static/doc checks | 5 Python files pass isort/black/ruff; 7 cookbook code blocks parse; CI shell syntax and diff whitespace pass |

The server subprocesses run from an artifact directory outside the checkout,
with PYTHONPATH and MOSS_SPEECH_VOICE_WAV removed. Only the installed executable
and CLI arguments supply serving configuration. This exercises actual native
inference, not a mocked server. CPU CLI tests mock only the launch boundary.

Initial GPU job3916 already passed all8 requests and shutdown, but its log exposed
the discovery prompt. Its report remains unchanged. The shared fix was applied
only after that job exited; new job3917 additionally requires no prompt. Runtime
source hashes and evidence receipts are in
sglang-omni/docs/design/moss_speech/cli_launch_summary.json.

The expanded asset-free CI entry includes the new tests, and its dependency list
explicitly includes Typer and Python3.10's TOML reader. Tests overlap and should
not be summed. This run uses the recorded P5/P6 dependency profile; it does not
qualify the newer default CUDA13 stack or change any numerical/hardware claim.
Historical P5/P6 reports and delivery manifests remain snapshots of their commits.
This correction is separate from P7, which remains unstarted.

## Developer reproduction

The installed console names must be on PATH. Run from the workspace root, inside
a Slurm GPU allocation with the existing offline environment and frozen assets:

```bash
python sglang-omni/scripts/moss_speech/validate_launch.py \
  --model-path models/MOSS-Speech --codec-path models/MOSS-Speech-Codec \
  --voice-wav repos/MOSS-Speech/assets/prompt-cn.wav \
  --streaming-config sglang-omni/examples/configs/moss_speech_streaming.yaml \
  --manifest artifacts/p5/seedtts/manifest.json \
  --reference artifacts/p5/reference_3880/report.json \
  --offline-baseline artifacts/p5/ci_3883 \
  --streaming-baseline artifacts/p6/ci_3903 \
  --out-dir artifacts/cli-new
```

This command is a developer regression driver; ordinary users use the direct
serve commands in the cookbook. Root Tasks.md, CHANGE.md, continue.md and raw
artifacts are local workspace records, outside the sglang-omni Git repository.
