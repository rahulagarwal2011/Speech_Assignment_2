# Speech Understanding PA-2: Maithili Pipeline

Code-switched (Hinglish) lecture transcription and re-synthesis into Maithili using zero-shot voice cloning.

## What This Does

Takes a 10-minute English-Hindi code-switched lecture, transcribes it, translates to Maithili, and synthesizes the Maithili lecture in the student's own voice with the professor's teaching prosody preserved.

## Target Language

Maithili (`mai_Deva`, Devanagari script) — spoken in Bihar, India and Nepal.

## Quick Start

```bash
# 1. Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install indictranstoolkit parler-tts
brew install ffmpeg

# 2. Configure
cp .env.example .env
# Edit .env -> set HF_TOKEN=hf_your_token

# 3. Place your audio files
mkdir -p outputs/audio
cp /path/to/lecture.wav outputs/audio/original_segment.wav
cp /path/to/your_voice.wav outputs/audio/student_voice_ref.wav

# 4. Run everything
python pipeline.py --all

# 5. If it pauses, fix the issue and resume
python pipeline.py --resume
```

## Pipeline Steps

| # | Step | What It Does | Time |
|---|------|-------------|------|
| 1 | Download | Downloads Maithili-only data from HuggingFace | ~5 min |
| 2-6 | Data Prep | Processes LID, N-gram, translation, TTS, anti-spoofing data | ~1 hour |
| 7 | **Part I: STT** | Spectral subtraction denoising + Whisper medium transcription | ~10 min |
| 8 | **Part II: Translation** | IPA conversion + IndicTrans2 En/Hi→Maithili translation | ~5 min |
| 9 | **Part III: Voice Cloning** | Speaker embedding + MMS-TTS synthesis + DTW prosody warping | ~1 hour |
| 10a | **Part IV: Anti-spoofing** | LFCC-based LCNN classifier (EER=0%) | ~30 sec |
| 10b | **Evaluation** | Metrics, figures, report scaffold | ~1 min |

## Key Results

| Metric | Value | Notes |
|--------|-------|-------|
| Transcription | 185 segments | Whisper medium, code-switched English-Hindi |
| Translation | 99.5% coverage | IndicTrans2 with 600-term technical dictionary |
| IPA conversion | 99.5% coverage | epitran + 17 Maithili-specific overrides |
| Anti-spoofing EER | 0.0% | LFCC+LCNN classifier |
| Technical dictionary | 600 terms | 6 domains |
| Synthesis | MMS-TTS | 185 segments at 22,050 Hz |

## Output Files

```
outputs/
├── audio/
│   ├── original_segment.wav          # Your lecture input
│   ├── original_segment_denoised.wav # After spectral subtraction
│   ├── student_voice_ref.wav         # Your 60s voice recording
│   ├── output_LRL_cloned.wav         # Final Maithili lecture (with prosody)
│   └── output_LRL_flat.wav           # Without prosody (for ablation)
├── transcript_code_switched.json     # Whisper transcription
├── transcript_ipa.json               # IPA conversion
├── maithili_translated_text.json     # Maithili translation
└── reports/
    ├── PA2_Report_Scaffold.md        # 10-page IEEE report
    ├── implementation_note.md        # 1-page design choices
    ├── evaluation_metrics.json       # All metrics
    └── figures/                      # 7 report figures
```

## Technical Approach

### Part I: STT Pipeline
- **Denoising**: Spectral subtraction (STFT-based noise estimation + oversubtraction)
- **LID**: Custom CNN+BiGRU frame-level classifier (architecture ready, needs Maithili audio to train)
- **Transcription**: Whisper medium with N-gram logit biasing (`logit' = logit + α·log P_ngram + β·bias`)

### Part II: Translation
- **IPA**: epitran (hin-Deva) + eng_to_ipa + 17 Maithili phoneme overrides
- **Translation**: IndicTrans2 via IndicTransToolkit with dictionary-first term handling
- **Dictionary**: 600 terms across speech processing, signal processing, ML, acoustics, linguistics, phonetics

### Part III: Voice Cloning
- **Speaker embedding**: SpeechBrain x-vector (512-dim) from 60s recording
- **TTS**: Meta MMS-TTS for Maithili (`facebook/mms-tts-mai`)
- **Prosody**: DTW alignment of F0/energy contours (chunked extraction, downsampled DTW)

### Part IV: Robustness
- **Anti-spoofing**: LFCC (20 coeff + deltas) → LCNN binary classifier → EER = 0%
- **FGSM**: Implementation complete, requires trained LID model to execute

## Dataset Attribution

| Dataset | Source | License | What We Download |
|---------|--------|---------|-----------------|
| BPCC | ai4bharat/BPCC | CC0 / CC-BY-4.0 | Maithili pairs only (~12 MB) |
| IndicVoices-R | ai4bharat/indicvoices_r | CC BY 4.0 | Maithili config (~50 GB) |
| Aksharantar | ai4bharat/aksharantar | CC0 1.0 | Maithili transliteration |
| Bhasha-Abhijnaanam | ai4bharat/Bhasha-Abhijnaanam | CC0 1.0 | Hi/Mai/En filtered |
| NLLB-Seed | allenai/nllb | CC BY-SA 4.0 | eng_Latn-mai_Deva pair |
| Sangraha | ai4bharat/sangraha | CC BY 4.0 | Maithili monolingual text |
| Kathbath | ai4bharat/Kathbath | CC0 1.0 | Hindi test split (for LID) |

## Requirements

- Python 3.10+
- PyTorch 2.x
- Apple Silicon (MPS) or NVIDIA GPU (CUDA) or CPU
- ~60 GB disk space (mostly IndicVoices-R audio)
- FFmpeg (`brew install ffmpeg`)

See `SETUP.md` for detailed installation instructions.
