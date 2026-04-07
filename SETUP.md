# Speech Understanding PA-2 — Setup and Execution Guide

## Target Language

Maithili (`mai_Deva`, Devanagari script)

## Constraints

- Python 3.10+ and PyTorch 2.x only
- All datasets must be free and open-source (CC BY 4.0, CC0, MIT, Apache 2.0)
- CUDA 12.x recommended for GPU acceleration (CPU works but is slower)

---

## Prerequisites

### 1. System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.10 | 3.10–3.11 |
| RAM | 16 GB | 32 GB |
| Disk space | 50 GB | 150 GB (with all datasets) |
| GPU | Not required | NVIDIA with 8+ GB VRAM |
| OS | macOS / Linux | Linux with CUDA 12.x |

### 2. Create a HuggingFace Account and Token

Several datasets and models are hosted on HuggingFace and require accepting license agreements.

1. Create an account at [huggingface.co](https://huggingface.co)
2. Go to [Settings > Access Tokens](https://huggingface.co/settings/tokens)
3. Create a new token with **Read** permissions
4. Accept the license agreements for these datasets by visiting each page and clicking "Agree":
   - [ai4bharat/samanantar](https://huggingface.co/datasets/ai4bharat/samanantar)
   - [ai4bharat/BPCC](https://huggingface.co/datasets/ai4bharat/BPCC)
   - [ai4bharat/kathbath](https://huggingface.co/datasets/ai4bharat/kathbath)
   - [ai4bharat/indicvoices_r](https://huggingface.co/datasets/ai4bharat/indicvoices_r)
   - [ai4bharat/aksharantar](https://huggingface.co/datasets/ai4bharat/aksharantar)
   - [ai4bharat/sangraha](https://huggingface.co/datasets/ai4bharat/sangraha)
   - [ai4bharat/Bhasha-Abhijnaanam](https://huggingface.co/datasets/ai4bharat/Bhasha-Abhijnaanam)
   - [ai4bharat/IndicVoices-ST](https://huggingface.co/datasets/ai4bharat/IndicVoices-ST)

### 3. Prepare Your Audio Files

You need **two audio files** before running the pipeline:

| File | Where to place it | Duration | Format | How to get it |
|------|--------------------|----------|--------|---------------|
| Lecture audio | `outputs/audio/original_segment.wav` | ~10 minutes | WAV, any sample rate | Extract a 10-min code-switched (Hinglish) segment from the provided class lectures |
| Your voice | `outputs/audio/student_voice_ref.wav` | Exactly 60 seconds | WAV, >= 22,050 Hz | Record yourself reading Maithili text in a quiet room |

The pipeline will pause and tell you if either file is missing when it reaches the step that needs it.

### 4. Install KenLM (Optional, for N-gram LM)

KenLM is used to build the ARPA language model. If you skip it, the pipeline falls back to frequency-based biasing.

```bash
# Ubuntu/Debian
sudo apt-get install libboost-all-dev cmake
pip install https://github.com/kpu/kenlm/archive/master.zip

# macOS
brew install boost cmake
pip install https://github.com/kpu/kenlm/archive/master.zip
```

---

## Installation

### Step 1: Clone the Repository

```bash
git clone <your-github-repo-url>
cd SpeechUnderstanding-PA2
```

### Step 2: Create Python Environment

**Option A — Conda (recommended)**

```bash
conda env create -f environment.yml
conda activate speech-pa2
```

**Option B — pip + venv**

```bash
python3.10 -m venv venv
source venv/bin/activate      # Linux/macOS
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2b: Additional Dependencies (Required)

These are needed but have version-specific requirements:

```bash
# IndicTrans2 translation (requires transformers 4.44.x)
pip install transformers==4.44.2
pip install indictranstoolkit

# Parler-TTS (optional, MMS is the default)
pip install parler-tts

# FFmpeg (required for audio decoding)
brew install ffmpeg        # macOS
# sudo apt install ffmpeg  # Ubuntu

# TorchCodec (required for HuggingFace audio datasets)
pip install torchcodec
```

### Step 3: Configure Environment Variables

```bash
cp .env.example .env
```

Open `.env` in your editor and fill in:

```dotenv
HF_TOKEN=hf_your_actual_token_here
TORCH_DEVICE=cuda                    # or cpu, or mps
```

All other values have sensible defaults. The full list:

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | *(required)* | HuggingFace access token for dataset downloads |
| `TORCH_DEVICE` | auto-detect | `cuda`, `cpu`, or `mps` (Apple Silicon) |
| `WHISPER_MODEL` | `large-v3` | Whisper model size for transcription |
| `TTS_MODEL` | `parler` | Primary TTS model (`parler` or `mms`) |
| `SPEAKER_EMBEDDING_METHOD` | `speechbrain` | Speaker embedding extractor (`speechbrain` or `resemblyzer`) |
| `LECTURE_AUDIO_PATH` | `outputs/audio/original_segment.wav` | Path to the lecture audio |
| `STUDENT_VOICE_PATH` | `outputs/audio/student_voice_ref.wav` | Path to your 60s voice recording |
| `LID_EPOCHS` | `30` | Training epochs for the LID model |
| `ANTISPOOF_EPOCHS` | `50` | Training epochs for the anti-spoofing classifier |
| `NGRAM_ALPHA` | `0.3` | Weight for N-gram LM in constrained decoding |
| `TECHNICAL_BETA` | `3.0` | Weight for technical vocabulary boosting |

### Step 4: Place Your Audio Files

```bash
mkdir -p outputs/audio
cp /path/to/your/lecture_segment.wav  outputs/audio/original_segment.wav
cp /path/to/your/voice_recording.wav  outputs/audio/student_voice_ref.wav
```

---

## Running the Pipeline

### One Command — Run Everything

```bash
python pipeline.py --all
```

This runs all 10 steps in order with progress tracking and automatic checkpointing.

### Resume After an Error

If the pipeline stops (missing file, network error, etc.), fix the issue and run:

```bash
python pipeline.py --resume
```

It skips already-completed steps and picks up where it left off.

### Run a Specific Step

```bash
python pipeline.py --step download          # just download datasets
python pipeline.py --step prepare_lid stt   # run specific steps
python pipeline.py --from stt               # run from Part I onward
```

### See What's Done

```bash
python pipeline.py --list
```

Shows all steps with checkmarks next to completed ones.

### Start Fresh

```bash
python pipeline.py --reset    # clear checkpoint
python pipeline.py --all      # run everything again
```

---

## Pipeline Steps in Detail

### Phase 1 — Data Acquisition (Step 1)

| Step | Name | What It Does |
|------|------|-------------|
| 1 | `download` | Downloads 9 datasets from HuggingFace into `data/raw/` |

Datasets downloaded (~100-150 GB total):

| Dataset | Size Estimate | Used For |
|---------|--------------|----------|
| Samanantar | ~500 MB | En-Mai parallel corpus (translation) |
| BPCC | ~2 GB | Gold-standard parallel pairs |
| NLLB-Seed | ~10 MB | High-quality validation pairs |
| Kathbath | ~5 GB | Maithili ASR audio + transcripts |
| IndicVoices-R | ~10-30 GB | TTS-quality Maithili speech (48kHz) |
| Aksharantar | ~100 MB | Transliteration pairs (G2P) |
| Sangraha | ~1 GB | Maithili monolingual text |
| Bhasha-Abhijnaanam | ~50 MB | Text LID data |
| IndicVoices-ST | ~5 GB | Speech translation triplets |

### Phase 2 — Data Preparation (Steps 2-6)

| Step | Name | Output |
|------|------|--------|
| 2 | `prepare_lid` | `data/processed/lid/` — spectrograms, frame labels, train/val/test splits |
| 3 | `prepare_ngram` | `data/processed/ngram/` — ARPA LM, technical vocab, logit bias JSON |
| 4 | `prepare_translation` | `data/processed/translation/` — parallel corpus, G2P table, IPA rules |
| 5 | `prepare_tts` | `data/processed/tts/` — TTS manifest, speaker embeddings, prosody features |
| 6 | `prepare_antispoof` | `data/processed/antispoofing/` — directory scaffold, LFCC pipeline |

### Phase 3 — Model Pipeline (Steps 7-10)

| Step | Name | Assignment Part | Output |
|------|------|----------------|--------|
| 7 | `stt` | Part I | `outputs/transcript_code_switched.json` — denoised, LID-tagged, transcribed lecture |
| 8 | `translation` | Part II | `outputs/maithili_translated_text.json` — IPA + Maithili translation |
| 9 | `voice_cloning` | Part III | `outputs/audio/output_LRL_cloned.wav` — 10-min synthesized Maithili lecture |
| 10a | `adversarial` | Part IV | `models/antispoof/` + adversarial examples |
| 10b | `evaluate` | Evaluation | `outputs/reports/` — metrics, figures, report scaffold |

---

## Output Files for Submission

After the pipeline completes, you will have:

### Required Audio Manifest

```
outputs/audio/original_segment.wav      ← Your input lecture
outputs/audio/student_voice_ref.wav     ← Your 60s voice reference
outputs/audio/output_LRL_cloned.wav     ← Final synthesized Maithili lecture
```

### Report Artifacts

```
outputs/reports/evaluation_metrics.json  ← All metrics with pass/fail
outputs/reports/PA2_Report_Scaffold.md   ← 10-page IEEE report template
outputs/reports/implementation_note.md   ← 1-page design choices template
outputs/reports/figures/                 ← 7 report figures (confusion matrix, etc.)
```

### Trained Models

```
models/lid/best_model.pt               ← LID model weights
models/antispoof/best_antispoof.pt     ← Anti-spoofing classifier weights
```

---

## Passing Criteria

The pipeline must meet these benchmarks:

| Metric | Threshold | Verified By |
|--------|-----------|-------------|
| WER (English segments) | < 15% | `evaluate` step |
| WER (Hindi segments) | < 25% | `evaluate` step |
| MCD (Mel-Cepstral Distortion) | < 8.0 dB | `evaluate` step |
| LID switching accuracy | within 200ms | `evaluate` step |
| Anti-spoofing EER | < 10% | `adversarial` step |
| Adversarial SNR | > 40 dB | `adversarial` step |
| Technical dictionary | >= 500 terms | `translation` step |

---

## Troubleshooting

### Known Issues on Apple Silicon (M4 Pro / M-series)

| Issue | Fix |
|-------|-----|
| `Cannot convert MPS Tensor to float64` | Whisper auto-runs on CPU (handled by pipeline) |
| `Unknown device for graph fuser` | TTS models auto-run on CPU (handled by pipeline) |
| `EncoderClassifier object has no attribute device_type` | SpeechBrain auto-runs on CPU (handled by pipeline) |
| `bus error` in Parselmouth/Praat | Prosody extraction uses 60s chunks (handled by code) |
| `torchcodec` library not loaded | Run `brew install ffmpeg` |
| `torchaudio.backend` not found (DeepFilterNet) | Falls back to spectral subtraction automatically |
| `transformers.onnx` module not found | Run `pip install transformers==4.44.2` |

### "HF_TOKEN not set"

```bash
# Check your .env file has the token
cat .env | grep HF_TOKEN

# Or export directly
export HF_TOKEN=hf_xxxxxxxxxxxxx
```

### "Lecture audio not found"

Place your 10-minute lecture WAV at `outputs/audio/original_segment.wav` and re-run with `--resume`.

### "Student voice recording not found"

Place your 60-second voice WAV at `outputs/audio/student_voice_ref.wav` and re-run with `--resume`.

### CUDA out of memory

Set a smaller Whisper model in `.env`:

```dotenv
WHISPER_MODEL=medium
```

Or run on CPU:

```dotenv
TORCH_DEVICE=cpu
```

### KenLM not found

The N-gram step will log a warning but won't crash. Install KenLM for full functionality (see Prerequisites above).

### A dataset fails to download

The pipeline logs the error and continues. You can re-run just the download step:

```bash
python scripts/download_datasets.py --dataset kathbath
```

Or download manually from the HuggingFace page and place in the corresponding `data/raw/<name>/` directory.

---

## Directory Structure

```
SpeechUnderstanding-PA2/
├── .env.example                ← Copy to .env and fill in your values
├── .env                        ← Your local config (gitignored)
├── .gitignore
├── requirements.txt
├── environment.yml
├── pipeline.py                 ← Single entry point: python pipeline.py --all
├── README.md
│
├── configs/
│   └── dataset_config.yaml     ← Central config for thresholds, paths, etc.
│
├── scripts/
│   ├── download_datasets.py    ← Step 1: download from HuggingFace
│   ├── prepare_lid_data.py     ← Step 2: LID data prep
│   ├── prepare_ngram_data.py   ← Step 3: N-gram LM prep
│   ├── prepare_translation_data.py ← Step 4: parallel corpus prep
│   ├── prepare_tts_data.py     ← Step 5: TTS data prep
│   ├── prepare_antispoof_data.py   ← Step 6: anti-spoof scaffold
│   ├── train_lid.py            ← LID model (CNN+BiGRU)
│   ├── constrained_decode.py   ← Whisper + N-gram logit biasing
│   ├── denoise_audio.py        ← DeepFilterNet / spectral subtraction
│   ├── run_stt_pipeline.py     ← Step 7: Part I orchestrator
│   ├── ipa_converter.py        ← Hinglish-to-IPA converter
│   ├── translate_to_maithili.py    ← IndicTrans2 translation
│   ├── build_technical_dictionary.py ← 500-word dictionary builder
│   ├── run_translation_pipeline.py  ← Step 8: Part II orchestrator
│   ├── extract_speaker_embedding.py ← x-vector / d-vector extraction
│   ├── prosody_warping.py      ← DTW prosody warping (custom, not wrapper)
│   ├── synthesize_maithili.py  ← Parler-TTS / MMS synthesis
│   ├── run_voice_cloning_pipeline.py ← Step 9: Part III orchestrator
│   ├── train_antispoof.py      ← LCNN anti-spoofing classifier
│   ├── adversarial_attack.py   ← FGSM adversarial noise injection
│   ├── run_adversarial_pipeline.py  ← Step 10a: Part IV orchestrator
│   ├── evaluate_all.py         ← Compute all metrics
│   ├── generate_figures.py     ← 7 report figures
│   ├── scaffold_report.py      ← IEEE report + implementation note
│   └── run_evaluation_pipeline.py   ← Step 10b: evaluation orchestrator
│
├── data/
│   ├── raw/                    ← Downloaded datasets (gitignored)
│   ├── processed/              ← Task-ready processed data
│   └── manual/
│       └── technical_parallel_corpus.tsv  ← 500-word dictionary
│
├── models/
│   ├── lid/                    ← Trained LID weights (gitignored)
│   └── antispoof/              ← Trained anti-spoof weights (gitignored)
│
└── outputs/
    ├── audio/                  ← Input + generated audio (gitignored)
    ├── reports/                ← Metrics, figures, report scaffold
    └── manifests/              ← Pipeline checkpoint + phase manifests
```

---

## Quick Start (TL;DR)

```bash
# 1. Install
conda env create -f environment.yml && conda activate speech-pa2

# 2. Configure
cp .env.example .env
# Edit .env → set HF_TOKEN=hf_your_token

# 3. Place audio files
mkdir -p outputs/audio
cp lecture.wav    outputs/audio/original_segment.wav
cp my_voice.wav   outputs/audio/student_voice_ref.wav

# 4. Run everything
python pipeline.py --all

# 5. If it stops, fix the issue and resume
python pipeline.py --resume
```
