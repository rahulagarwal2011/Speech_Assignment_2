"""
Part IV orchestrator: adversarial robustness and anti-spoofing pipeline.

Ties together Task 4.1 (anti-spoofing) and Task 4.2 (adversarial attack):
  1. Segment bonafide and spoof audio into clips
  2. Train LCNN anti-spoofing classifier, evaluate EER
  3. Select a 5 s Hindi segment for adversarial attack
  4. Load the Phase 3 LID model
  5. Run FGSM with binary search for minimum epsilon
  6. Save all artefacts and a summary JSON

Usage:
  python scripts/run_adversarial_pipeline.py \
      --config configs/dataset_config.yaml

  Or with explicit paths:
  python scripts/run_adversarial_pipeline.py \
      --bonafide-audio outputs/audio/student_voice_ref.wav \
      --spoof-audio outputs/audio/output_LRL_cloned.wav \
      --lid-model models/lid/best_model.pt
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Hindi segment selection
# ---------------------------------------------------------------------------

def select_hindi_segment(
    audio_path: str,
    lid_model_path: str,
    duration_sec: float = 5.0,
    sr: int = 22050,
    device: str = "cpu",
) -> Optional[np.ndarray]:
    """Pick the best 5 s Hindi segment from the denoised lecture audio.

    Strategy: slide a window across the audio, run LID on each window, and
    choose the window whose frames are most consistently predicted as Hindi.
    Falls back to the first 5 s if no strong Hindi region is found.
    """
    import soundfile as sf
    import librosa
    import torch

    from scripts.adversarial_attack import load_lid_model, audio_to_logmel
    from scripts.train_lid import LANG_MAP

    audio_np, file_sr = sf.read(audio_path, dtype="float32")
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    if file_sr != sr:
        audio_np = librosa.resample(audio_np, orig_sr=file_sr, target_sr=sr)

    seg_len = int(duration_sec * sr)
    if len(audio_np) < seg_len:
        logger.warning(
            f"Audio shorter than {duration_sec}s — using full audio, zero-padded"
        )
        padded = np.zeros(seg_len, dtype=np.float32)
        padded[: len(audio_np)] = audio_np
        return padded

    model = load_lid_model(lid_model_path, device)
    hi_label = LANG_MAP["hi"]

    hop = sr  # 1 s hop for candidate windows
    best_score = -1.0
    best_start = 0

    for start in range(0, len(audio_np) - seg_len + 1, hop):
        chunk = audio_np[start : start + seg_len]
        waveform = torch.from_numpy(chunk).float()
        log_mel = audio_to_logmel(waveform, sr=sr).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(log_mel)
        preds = logits.argmax(dim=-1).squeeze(0)
        hi_frac = (preds == hi_label).float().mean().item()

        if hi_frac > best_score:
            best_score = hi_frac
            best_start = start

    logger.info(
        f"Selected Hindi segment at {best_start / sr:.2f}s "
        f"(Hindi frame fraction: {best_score:.2%})"
    )
    return audio_np[best_start : best_start + seg_len].copy()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_adversarial_pipeline(
    config_path: str = "configs/dataset_config.yaml",
    bonafide_audio: str = "outputs/audio/student_voice_ref.wav",
    spoof_audio: str = "outputs/audio/output_LRL_cloned.wav",
    lid_model_path: str = "models/lid/best_model.pt",
    lecture_audio: str = "outputs/audio/original_segment_denoised.wav",
    output_dir: str = "outputs",
    antispoof_model_dir: str = "models/antispoof",
    epochs: int = 50,
    device: str = None,
) -> Dict:
    """Run the full Part IV pipeline.

    Returns a summary dict with EER, epsilon, SNR, and file paths.
    """
    import torch
    import soundfile as sf

    if device is None:
        from scripts.device_utils import get_device
        device = get_device()

    config = load_config(config_path) if Path(config_path).exists() else {}
    sr = config.get("audio", {}).get("target_sample_rate", 22050)
    eer_threshold = config.get("evaluation", {}).get("eer_threshold", 0.10)
    snr_min = config.get("evaluation", {}).get("adversarial_snr_min_db", 40)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "audio").mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    summary: Dict = {"pipeline": "adversarial-antispoof"}

    # ------------------------------------------------------------------
    # Task 4.1  — Anti-spoofing
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("TASK 4.1: Train LCNN Anti-Spoofing Classifier")
    logger.info("=" * 60)

    from scripts.train_antispoof import train_antispoof

    antispoof_results = train_antispoof(
        bonafide_audio=bonafide_audio,
        spoof_audio=spoof_audio,
        output_dir=antispoof_model_dir,
        epochs=epochs,
        device=device,
    )

    test_eer = antispoof_results["test_eer"]
    summary["antispoof"] = {
        "test_eer": test_eer,
        "eer_threshold": eer_threshold,
        "pass": test_eer < eer_threshold,
        "model_path": str(Path(antispoof_model_dir) / "best_antispoof.pt"),
    }
    logger.info(f"Anti-spoofing EER: {test_eer:.4f} (target < {eer_threshold})")

    # ------------------------------------------------------------------
    # Task 4.2  — FGSM adversarial attack
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("TASK 4.2: FGSM Adversarial Attack on LID")
    logger.info("=" * 60)

    source_for_segment = lecture_audio
    if not Path(lecture_audio).exists():
        logger.warning(
            f"Denoised lecture not found at {lecture_audio}; "
            f"falling back to spoof audio for Hindi segment selection"
        )
        source_for_segment = spoof_audio

    hindi_segment = select_hindi_segment(
        source_for_segment, lid_model_path, duration_sec=5.0, sr=sr, device=device,
    )

    orig_path = out / "audio" / "adversarial_original_5s.wav"
    sf.write(str(orig_path), hindi_segment, sr)
    logger.info(f"Original 5 s segment saved to {orig_path}")

    from scripts.adversarial_attack import run_attack

    attack_results = run_attack(
        audio_path=str(orig_path),
        lid_model_path=lid_model_path,
        target_label_str="en",
        output_dir=output_dir,
        device=device,
    )

    summary["adversarial"] = {
        "flip_achieved": attack_results["flip_achieved"],
        "min_epsilon": attack_results.get("min_epsilon"),
        "snr_at_min_db": attack_results.get("snr_at_min_db"),
        "snr_threshold_db": snr_min,
        "snr_pass": (
            attack_results.get("snr_at_min_db") is not None
            and attack_results["snr_at_min_db"] >= snr_min
        ),
        "original_segment_path": str(orig_path),
        "adversarial_audio_path": attack_results.get("adversarial_audio_path"),
    }

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    summary["elapsed_sec"] = round(elapsed, 1)

    summary_path = out / "part4_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Pipeline summary saved to {summary_path}")

    logger.info("=" * 60)
    logger.info("PART IV COMPLETE")
    logger.info(f"  Anti-spoof EER : {test_eer:.4f} {'PASS' if test_eer < eer_threshold else 'FAIL'}")
    if attack_results["flip_achieved"]:
        logger.info(
            f"  Adversarial eps : {attack_results['min_epsilon']:.6f}  "
            f"SNR={attack_results['snr_at_min_db']:.1f} dB"
        )
    else:
        logger.info("  Adversarial     : flip NOT achieved")
    logger.info(f"  Total time      : {elapsed:.1f}s")
    logger.info("=" * 60)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Part IV: adversarial robustness and anti-spoofing pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset_config.yaml",
    )
    parser.add_argument(
        "--bonafide-audio",
        type=str,
        default="outputs/audio/student_voice_ref.wav",
    )
    parser.add_argument(
        "--spoof-audio",
        type=str,
        default="outputs/audio/output_LRL_cloned.wav",
    )
    parser.add_argument(
        "--lid-model",
        type=str,
        default="models/lid/best_model.pt",
    )
    parser.add_argument(
        "--lecture-audio",
        type=str,
        default="outputs/audio/original_segment_denoised.wav",
        help="Denoised lecture audio for Hindi segment selection",
    )
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--antispoof-model-dir", type=str, default="models/antispoof")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    summary = run_adversarial_pipeline(
        config_path=args.config,
        bonafide_audio=args.bonafide_audio,
        spoof_audio=args.spoof_audio,
        lid_model_path=args.lid_model,
        lecture_audio=args.lecture_audio,
        output_dir=args.output_dir,
        antispoof_model_dir=args.antispoof_model_dir,
        epochs=args.epochs,
        device=args.device,
    )
