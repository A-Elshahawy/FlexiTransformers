# FlexiTransformers Examples

Each file is self-contained and runnable with `python examples/<file>.py`.

| File | What it covers |
|------|---------------|
| [01_quick_start.py](01_quick_start.py) | `FlexiGPT`, `FlexiBERT`, `FlexiTransformer` convenience constructors |
| [02_manual_config.py](02_manual_config.py) | `ModelConfig` + `create_model` — all three architectures |
| [03_positional_encodings.py](03_positional_encodings.py) | Every PE type end-to-end; direct PE object construction; `create_pe` factory |
| [04_encoder_only.py](04_encoder_only.py) | BERT-style encoder — built-in head, `BertHead`, `SequenceClassificationHead`, `TokenClassificationHead` |
| [05_encoder_decoder.py](05_encoder_decoder.py) | T5/BART seq2seq — `Batch`, `LabelSmoothing`, `LossCompute`, `run_epoch` |
| [06_decoder_only.py](06_decoder_only.py) | GPT-style LM — training loop + `greedy_decode` generation |
| [07_output_heads.py](07_output_heads.py) | `LMHead` with weight tying, all four head types, activation variants |
| [08_save_load.py](08_save_load.py) | `model.save()` / `ModelClass.load()` — decoder-only and encoder-only |
| [09_custom_pe.py](09_custom_pe.py) | Custom PE plugin via `register_pe` — `NoPE` and `ScaledSinusoidal` examples |
