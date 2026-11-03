# MOSS-Speech canonical processor inputs

canonical_inputs.json contains five frozen processor inputs: text-to-text, text-to-speech, speech-to-text, speech-to-speech, and mixed multi-turn history. It records the normalized conversations, exact two-channel grids, attention masks, and fixed tokenization of the text segments used by those conversations. The source revisions and SHA256 values are recorded in the fixture itself.

The reference interface's implicit system prompts are made explicit in these conversations before the processor is called. Audio source paths are normalized to placeholders; their recovered code sequences and grids are unchanged. Separate cases exercise the production processor's own default system prompts, so the frozen interface inputs do not stand in for that behavior.

The inputs originate from MOSS-Speech revision cff025bb41d8459d59abac0b5e44aba7f659ec9e, codec revision eeec733e4e1dea7da444d332d8e1621ef257414c, and reference revision 1ea408a10d07b9fdc7a27bce19d1211b62067784. The preserved source files were read from commit 7e8ab1055efcd7f7c295a27540899d1e2d5845e5.

Each audio-input case contains the complete 343 codec codes. Those codes were recovered from the canonical grid between the start row [151646,512] and end row [151667,16384], verifying that every intervening row has text token 151667 and a valid audio code. The three recovered sequences are identical. The source metadata contains only a truncated preview and was not used as a complete code sequence.

The fixture tokenizer substitutes only the external tokenization boundary: known text segments return their frozen reference IDs, while unrelated test strings use a deterministic UTF-8 mapping. The production processor constructs every tested grid and padded batch. These tests do not validate the real tokenizer vocabulary or regenerate codec codes; real checkpoint and audio inference belong to the accelerator/service validation.

Only processor inputs are retained here. Historical FP32 generated grids, logits, and audio outputs are not native BF16 generation expectations. No test requires an external alignment directory, a model cache, or a private workspace path.
