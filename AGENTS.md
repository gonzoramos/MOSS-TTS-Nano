# AGENTS.md

## What this repo is

MOSS-TTS-Nano is a 0.1B-parameter multilingual TTS model (voice cloning). It generates 48 kHz stereo audio on CPU via a pure autoregressive pipeline: text normalization → TTS LLM (RVQ codes) → Audio Tokenizer (waveform).

No test suite, no CI, no linter config. Research/demo codebase.

---

## Setup (order matters)

```bash
conda create -n moss-tts-nano python=3.12 -y
conda activate moss-tts-nano
# pynini MUST come before WeTextProcessing
conda install -c conda-forge pynini=2.1.6.post1 -y
pip install git+https://github.com/WhizZest/WeTextProcessing.git
pip install -r requirements.txt
pip install -e .
```

`pynini` cannot be reliably installed via pip on most platforms — use the conda-forge build.

---

## Running

```bash
# CLI inference
python infer.py --prompt-audio-path assets/audio/zh_1.wav --text "..."

# Web demo (FastAPI, port 18083)
python app.py

# Packaged CLI (after pip install -e .)
moss-tts-nano generate --prompt-speech assets/audio/zh_1.wav --text "..."
moss-tts-nano serve

# Pre-clone a default voice (one-time; encodes RVQ codes into cache)
moss-tts-nano set-voice --prompt-speech assets/audio/en_kayla.wav

# Generate using the pre-cloned default voice (no --prompt-speech needed)
moss-tts-nano generate --text "..."
```

---

## Entrypoints

| File | Role |
|---|---|
| `infer.py` | Direct CLI inference |
| `app.py` | FastAPI server + embedded HTML/JS UI (1600+ lines) |
| `moss_tts_nano/cli.py` | Packaged CLI (`generate` / `serve` / `set-voice`) |
| `moss_tts_nano/__main__.py` | `python -m moss_tts_nano` |
| `moss_tts_nano/config.py` | Read/write user config at `~/.config/moss-tts-nano/config.json` |
| `moss_tts_nano/voice_cache.py` | Pre-encode reference audio → RVQ codes; cache inject at inference time |
| `moss_tts_nano_runtime.py` | `NanoTTSService` — the core runtime used by `app.py` |
| `finetuning/sft.py` | Supervised finetuning (Accelerate + AdamW) |
| `finetuning/prepare_data.py` | Precomputes RVQ audio codes before training |
| `finetuning/verify.py` | Post-finetune quick inference check |

---

## Architecture

- **`NanoTTSService`** (`moss_tts_nano_runtime.py`): central runtime. Lazy-loads both models with `threading.RLock`. Auto-selects device/dtype/attention backend (flash_attention_2 if available, else sdpa, else eager).
- **TTS model**: `OpenMOSS-Team/MOSS-TTS-Nano` loaded with `trust_remote_code=True`. Has a global Transformer + `local_transformer`, 16 RVQ codebooks, text + audio LM heads.
- **Audio tokenizer**: `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano` (~20M params). Decodes RVQ codes to 48 kHz stereo waveform.
- **Text normalization**: two-layer — regex via `tts_robust_normalizer_single_script.py`, then WeTextProcessing (lazy-loaded in background thread). WeTextProcessing cache goes to `.cache/wetext_zh_no_erhua_keep_punct/`.

---

## Models

Downloaded automatically from HuggingFace on first run. For offline use pass `--checkpoint` and `--audio-tokenizer` with local paths.

- `OpenMOSS-Team/MOSS-TTS-Nano`
- `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano`

---

## Finetuning

```bash
# 1. Precompute audio codes (must run before training)
python finetuning/prepare_data.py \
  --codec-path ./models/MOSS-Audio-Tokenizer-Nano \
  --input-jsonl train_raw.jsonl \
  --output-jsonl train_with_codes.jsonl

# 2. Train
accelerate launch finetuning/sft.py \
  --model-path ./models/MOSS-TTS-Nano \
  --codec-path ./models/MOSS-Audio-Tokenizer-Nano \
  --train-jsonl train_with_codes.jsonl \
  --output-dir output/moss_tts_nano_sft \
  --per-device-batch-size 1 --gradient-accumulation-steps 8 \
  --learning-rate 1e-5 --num-epochs 3 --mixed-precision bf16

# Or use the one-click wrapper (reads env vars)
bash finetuning/run_train.sh
```

Key finetuning env vars (for `run_train.sh`): `MODEL_PATH`, `CODEC_PATH`, `RAW_JSONL`, `PREPARED_JSONL`, `OUTPUT_DIR`, `SKIP_PREPARE` (set to `1` to skip data prep).

GPU memory: ~3.23 GiB at batch=1, max_length=1024, bf16.

---

## Troubleshooting (Poetry/macOS)

- If `poetry install` fails with `pyproject.toml changed significantly since poetry.lock was last generated`, run:

```bash
poetry lock
poetry install
```

- If inference fails with `ModuleNotFoundError: No module named 'torch'`, the Poetry env is missing deps. Re-run `poetry install`.

- If inference fails with `RuntimeError: Couldn't find appropriate backend to handle uri ...` from `torchaudio.load(...)`, install system audio backends:

```bash
brew install ffmpeg sox
```

Then rerun the same `poetry run python infer.py ...` command.

---

## Output paths

- Generated audio: `generated_audio/`
- App prompt uploads: `.app_prompt_uploads/`
- WeTextProcessing cache: `.cache/wetext_zh_no_erhua_keep_punct/`
- Default voice config: `~/.config/moss-tts-nano/config.json`
- Pre-encoded RVQ code cache: `~/.config/moss-tts-nano/cache/<sha256>.pt`
