# Vendored MOSS-Speech codec sources

This subtree contains the inference dependencies used by the serving adapter in `sglang_omni/models/moss_speech/components/`. It retains checkpoint module names and eager computation order. No weights, native kernels, graph owners, prepared caches or local attention processors are bundled.

| Source | Fixed revision | Scope |
|---|---|---|
| OpenMOSS-Team/MOSS-Speech-Codec, Hugging Face | `eeec733e4e1dea7da444d332d8e1621ef257414c` | Whisper-VQ configuration/encoder/token extraction and explicit flow/HiFT construction |
| OpenMOSS/MOSS-Speech, GitHub | `1ea408a10d07b9fdc7a27bce19d1211b62067784` | CosyVoice flow, HiFT, encoder and utility inference closure |
| shivammehta25/Matcha-TTS | `bd4d90d93214b37f7a159cf205ae85762c2c10aa` | Mel extraction and layers used by the flow estimator |

## License evidence and remaining gate

The [pinned Hugging Face API](https://huggingface.co/api/models/OpenMOSS-Team/MOSS-Speech-Codec/revision/eeec733e4e1dea7da444d332d8e1621ef257414c), checked on 2026-10-27, lists no `README.md` or license file, and provides no model-card license metadata. The old `fnlp/MOSS-Speech-Codec` API URL redirects to `OpenMOSS-Team/MOSS-Speech-Codec` with the same repository ID and exact revision; the namespace change does not supply additional license terms.

The pinned HF [`modeling_moss_speech_codec.py`](https://huggingface.co/OpenMOSS-Team/MOSS-Speech-Codec/blob/eeec733e4e1dea7da444d332d8e1621ef257414c/modeling_moss_speech_codec.py) explicitly carries an Apache-2.0 header, with copyright attributed to the 2025 OpenMOSS and HuggingFace Inc. teams. Its original header is retained in `hf_codec/modeling.py`, and the full license text is in `hf_codec/LICENSE.Apache-2.0`. The source `configuration_moss_speech_codec.py` also has this header, but is not copied into this subtree. These file-specific declarations do not establish terms for separate headerless source files or weights.

The pinned HF [`utils.py`](https://huggingface.co/OpenMOSS-Team/MOSS-Speech-Codec/blob/eeec733e4e1dea7da444d332d8e1621ef257414c/utils.py), [`modeling_whisper.py`](https://huggingface.co/OpenMOSS-Team/MOSS-Speech-Codec/blob/eeec733e4e1dea7da444d332d8e1621ef257414c/modeling_whisper.py) and [`flow/config.yaml`](https://huggingface.co/OpenMOSS-Team/MOSS-Speech-Codec/blob/eeec733e4e1dea7da444d332d8e1621ef257414c/flow/config.yaml) contain no license or copyright statement. They supply `hf_codec/utils.py`, `hf_codec/whisper.py` and constructor configuration in `hf_codec/modeling.py`. Applicable redistribution terms for those portions remain unconfirmed. Before publishing this vendor subtree, obtain an explicit applicable grant for these sources or replace them with sources whose compatible terms are verified. `hf_codec/LICENSE.note` records the gap and is not a license grant.

The pinned GitHub MOSS-Speech [README](https://github.com/OpenMOSS/MOSS-Speech/blob/1ea408a10d07b9fdc7a27bce19d1211b62067784/README.md#license) declares Apache-2.0 for code in that repository; its [license text](https://github.com/OpenMOSS/MOSS-Speech/blob/1ea408a10d07b9fdc7a27bce19d1211b62067784/LICENSE) accompanies `cosyvoice_flow/`. Original per-file copyright and license notices are retained, including the restored notice in the trimmed `utils_common.py`; modified Apache-derived files identify the local changes through this document. The Matcha-TTS [MIT license](https://github.com/shivammehta25/Matcha-TTS/blob/bd4d90d93214b37f7a159cf205ae85762c2c10aa/LICENSE), copyright 2023 Shivam Mehta, is retained byte-for-byte in `matcha_components/LICENSE`.

CosyVoice files also explicitly attribute portions to other projects. Their associated MIT notices are retained below. The inspected revisions pin the notice evidence; MOSS-Speech does not identify the original copied revision for these portions, so these hashes are not asserted to be their exact code origins.

| Attributed source | Inspected notice revision | Local notice and attributed code |
|---|---|---|
| [EdwardDixon/snake](https://github.com/EdwardDixon/snake/blob/90343ed9458364dc3e622fc37be49d35afc26a6b/LICENSE) | `90343ed9458364dc3e622fc37be49d35afc26a6b` | `cosyvoice_flow/transformer/LICENSE.snake`; `activation.py` Snake |
| [jik876/hifi-gan](https://github.com/jik876/hifi-gan/blob/4769534d45265d52a904b850da5a622601885777/LICENSE) | `4769534d45265d52a904b850da5a622601885777` | `cosyvoice_flow/LICENSE.hifi-gan`; `hifigan_generator.py` |
| [kan-bayashi/ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN/blob/86740373ec609cb9fb192d472d2aea125041491a/LICENSE) | `86740373ec609cb9fb192d472d2aea125041491a` | `cosyvoice_flow/LICENSE.parallel-wavegan`; `hifigan_generator.py` |
| [NVIDIA/BigVGAN](https://github.com/NVIDIA/BigVGAN/blob/7d2b454564a6c7d014227f635b7423881f14bdac/LICENSE) | `7d2b454564a6c7d014227f635b7423881f14bdac` | `cosyvoice_flow/LICENSE.bigvgan`; `hifigan_generator.py` |
| [OpenAI Whisper](https://github.com/openai/whisper/blob/c0d2f624c09dc18e709e37c2ad90c039a4eb72a2/LICENSE) | `c0d2f624c09dc18e709e37c2ad90c039a4eb72a2` | `cosyvoice_flow/transformer/LICENSE.whisper`; `subsampling.py` Conv1dSubsampling2 |

The MoEFFN source references in `positionwise_feed_forward.py` and the ESPnet source references remain intact. The inspected [LitGPT](https://github.com/Lightning-AI/litgpt/blob/b3e83b8a22d35f676edcaf51f721f0a2e5f28bfa/LICENSE) and [Mistral](https://github.com/mistralai/mistral-inference/blob/6eb35510403825cfb430b0004443053e8c4b70dc/LICENSE) notices are Apache-2.0; their inspected commits likewise do not establish the exact revisions copied by CosyVoice.

No weights are bundled here. Checkpoint terms remain separate from code terms: the official repository's [checkpoint-license issue](https://github.com/OpenMOSS/MOSS-Speech/issues/3) was open with no comments at the check date, and supplies no maintainer grant. Neither the GitHub code declaration nor any license in SGLang-Omni is treated as a checkpoint license.

## Source mapping and modifications

All imports are rewired into this vendor namespace. Methods outside the production inference closure are omitted where listed below. Library compatibility changes are recorded separately from numerical computation changes.

| Local path | Source path | Modifications |
|---|---|---|
| `hf_codec/utils.py` | HF `utils.py` | Retains WhisperVQConfig, token extraction and resampling only; path audio uses soundfile to return float32 channel-first samples |
| `hf_codec/whisper.py` | HF `modeling_whisper.py` | Retains encoder/VQ/causal inference closure; guards removed Transformers legacy-cache constructor; reads return_dict directly with the TorchScript tuple fallback |
| `hf_codec/modeling.py` | HF `modeling_moss_speech_codec.py`, `flow/config.yaml` | Explicit frozen module construction replaces training YAML evaluation; construction saves/restores RNG; AudioDecoder retains the complete-sequence tensor-conditioned path; unused HF facade, path prompt API, duplicate speaker session and intermediate per-UUID decoder caches are omitted |
| `cosyvoice_flow/flow.py` | `cosyvoice/flow/flow.py` | Retains CausalMaskedDiffWithXvec inference and checkpoint attributes; unused noncausal model, training forward and training configuration dictionaries are omitted |
| `cosyvoice_flow/flow_matching.py` | `cosyvoice/flow/flow_matching.py` | Retains CausalConditionalCFM, shared Euler solver and fixed seed-0 noise; unused noncausal forward, training loss and TensorRT estimator branch are omitted |
| `cosyvoice_flow/decoder.py` | `cosyvoice/flow/decoder.py` | Retains causal estimator and its layers; removes unused noncausal base, with the causal class directly inheriting nn.Module and preserving its explicit constructor, inherited initialize_weights implementation and state-dict names |
| `cosyvoice_flow/transformer/attention.py` | `cosyvoice/transformer/attention.py` | Eager source restored; no storage-reuse or relative-shift view optimization |
| `cosyvoice_flow/hifigan_generator.py` | `cosyvoice/hifigan/generator.py` | Import rewiring; mutable constructor defaults use immutable dilation tuples or fresh lists/dictionaries with the same values; explicit checkpoint construction is unchanged |
| `cosyvoice_flow/hifigan_f0_predictor.py` | `cosyvoice/hifigan/f0_predictor.py` | Import rewiring only |
| `cosyvoice_flow/transformer/activation.py`, `convolution.py`, `embedding.py`, `encoder_layer.py`, `positionwise_feed_forward.py`, `subsampling.py`, `upsample_encoder.py` | Corresponding `cosyvoice/transformer/*.py` | Import rewiring only |
| `cosyvoice_flow/class_utils.py` | `cosyvoice/utils/class_utils.py` | Retains activation/subsample/position/attention registries needed by UpsampleConformerEncoder; removes unrelated model factories |
| `cosyvoice_flow/utils_common.py` | `cosyvoice/utils/common.py` | Retains seed, mask-bias, padding and initialization helpers only |
| `cosyvoice_flow/utils_mask.py` | `cosyvoice/utils/mask.py` | Import rewiring only |
| `matcha_components/audio.py` | `matcha/utils/audio.py` | Import rewiring only; preserves mel computation |
| `matcha_components/decoder.py` | `matcha/models/components/decoder.py` | Retains layers needed by the causal estimator; removes unused full decoder and Conformer wrapper |
| `matcha_components/transformer.py` | `matcha/models/components/transformer.py` | Import rewiring only; uses installed Diffusers attention |
| `matcha_components/flow_matching_base.py` | `matcha/models/components/flow_matching.py` | Retains BASECFM inference initialization and Euler methods; removes training and unused dependencies |

The offline `flow.pt` constructor retains its frozen chunk-size value of 5; offline attention uses the full context. Streaming uses the explicitly selected `flow-chunk-{chunk_size}.pt` checkpoint and matching chunk masks. Request RNG isolation, HiFT overlap and final sample accounting are owned by the serving streaming codec, outside this subtree. Voice prompt codes are encoded from the original input sample rate; mel features use 24 kHz audio and the speaker extractor uses its 24-to-16 kHz resample chain.
