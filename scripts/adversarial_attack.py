"""
FGSM adversarial attack on the frame-level LID model for Task 4.2.

Performs a white-box Fast Gradient Sign Method attack to craft adversarial
audio that causes the LID model to misclassify a Hindi speech segment as
English, while keeping the perturbation imperceptible (SNR > 40 dB).

Includes binary search over epsilon to find the minimum perturbation that
flips the model's prediction.

Expects:
  - Trained LID model at models/lid/best_model.pt
  - A short (≈5 s) Hindi speech segment as WAV

Usage:
  python scripts/adversarial_attack.py \
      --audio outputs/audio/adversarial_original_5s.wav \
      --lid-model models/lid/best_model.pt \
      --target-label en \
      --output-dir outputs/
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Import LID model definition from Phase 3
# ---------------------------------------------------------------------------

from scripts.train_lid import FrameLevelLID, LANG_MAP, LANG_MAP_INV, NUM_CLASSES


# ---------------------------------------------------------------------------
# Audio ↔ log-mel spectrogram (differentiable via torchaudio)
# ---------------------------------------------------------------------------

def audio_to_logmel(
    waveform: torch.Tensor,
    sr: int = 22050,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 220,
    win_length: int = 440,
) -> torch.Tensor:
    """Convert raw waveform to log-mel spectrogram (differentiable).

    Args:
        waveform: (1, num_samples) or (num_samples,) float tensor.
        sr: Sample rate.

    Returns:
        (n_mels, T) log-mel spectrogram.
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
    ).to(waveform.device)

    spec = mel_spec(waveform)  # (1, n_mels, T)
    log_spec = torch.log(spec.clamp(min=1e-9))
    return log_spec.squeeze(0)  # (n_mels, T)


# ---------------------------------------------------------------------------
# SNR utility
# ---------------------------------------------------------------------------

def compute_snr(original: torch.Tensor, perturbation: torch.Tensor) -> float:
    """Compute Signal-to-Noise Ratio in dB.

    SNR = 10 * log10( sum(original^2) / sum(perturbation^2) )
    """
    sig_power = (original.float() ** 2).sum().item()
    noise_power = (perturbation.float() ** 2).sum().item()
    if noise_power < 1e-20:
        return float("inf")
    return 10.0 * np.log10(sig_power / noise_power)


# ---------------------------------------------------------------------------
# Majority-vote prediction helper
# ---------------------------------------------------------------------------

def _predict_majority(
    model: nn.Module,
    waveform: torch.Tensor,
    device: str,
    sr: int = 22050,
) -> int:
    """Run LID model on waveform and return majority-vote language label."""
    model.eval()
    log_mel = audio_to_logmel(waveform, sr=sr)  # (80, T)
    log_mel = log_mel.unsqueeze(0).to(device)    # (1, 80, T)
    with torch.no_grad():
        logits = model(log_mel)                  # (1, T, 3)
    preds = logits.argmax(dim=-1).squeeze(0)     # (T,)
    majority = int(torch.mode(preds).values.item())
    return majority


# ---------------------------------------------------------------------------
# FGSM attack
# ---------------------------------------------------------------------------

def fgsm_attack(
    model: nn.Module,
    audio_tensor: torch.Tensor,
    target_label: int,
    epsilon: float,
    device: str = "cpu",
    sr: int = 22050,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply one-step FGSM perturbation targeting *target_label*.

    This is a **targeted** FGSM: we minimise the cross-entropy loss w.r.t.
    the target class so the model output shifts toward it.

    Args:
        model: Trained FrameLevelLID model.
        audio_tensor: 1-D float waveform tensor (will be cloned).
        target_label: Integer class index to fool the model into predicting.
        epsilon: Perturbation magnitude.
        device: Torch device.
        sr: Sample rate.

    Returns:
        (adversarial_audio, perturbation) both on CPU.
    """
    model.eval()

    adv_input = audio_tensor.clone().detach().to(device).requires_grad_(True)

    log_mel = audio_to_logmel(adv_input, sr=sr)  # (80, T)
    log_mel = log_mel.unsqueeze(0)                # (1, 80, T)

    logits = model(log_mel)                       # (1, T, C)
    T = logits.shape[1]

    target = torch.full((1, T), target_label, dtype=torch.long, device=device)
    loss = F.cross_entropy(logits.reshape(-1, NUM_CLASSES), target.reshape(-1))

    loss.backward()

    grad_sign = adv_input.grad.sign()
    perturbation = -epsilon * grad_sign
    adversarial = (adv_input + perturbation).clamp(-1.0, 1.0)

    return adversarial.detach().cpu(), perturbation.detach().cpu()


# ---------------------------------------------------------------------------
# Binary search for minimum epsilon
# ---------------------------------------------------------------------------

def find_minimum_epsilon(
    model: nn.Module,
    audio_tensor: torch.Tensor,
    original_label: int,
    target_label: int,
    device: str = "cpu",
    sr: int = 22050,
    eps_low: float = 0.0001,
    eps_high: float = 0.1,
    max_iters: int = 20,
    snr_min_db: float = 40.0,
) -> Dict:
    """Binary search for the minimum epsilon that flips the LID prediction.

    At each step we also check that SNR stays above snr_min_db.

    Returns:
        Dictionary with min_epsilon, snr_at_min, flipped, tradeoff_curve.
    """
    tradeoff: List[Dict] = []
    best_eps: Optional[float] = None
    best_snr: Optional[float] = None
    best_adv: Optional[torch.Tensor] = None

    for i in range(max_iters):
        eps_mid = (eps_low + eps_high) / 2.0

        adv, pert = fgsm_attack(model, audio_tensor, target_label, eps_mid, device, sr)
        pred = _predict_majority(model, adv, device, sr)
        snr = compute_snr(audio_tensor, pert)
        flipped = pred == target_label

        tradeoff.append({
            "iteration": i,
            "epsilon": eps_mid,
            "snr_db": snr,
            "flipped": flipped,
            "predicted_label": LANG_MAP_INV.get(pred, str(pred)),
        })
        logger.info(
            f"  iter={i} eps={eps_mid:.6f} snr={snr:.1f}dB "
            f"flipped={flipped} pred={LANG_MAP_INV.get(pred, pred)}"
        )

        if flipped:
            if best_eps is None or eps_mid < best_eps:
                best_eps = eps_mid
                best_snr = snr
                best_adv = adv
            eps_high = eps_mid
        else:
            eps_low = eps_mid

    result: Dict = {
        "min_epsilon": best_eps,
        "snr_at_min_db": best_snr,
        "flip_achieved": best_eps is not None,
        "snr_above_threshold": (best_snr is not None and best_snr >= snr_min_db),
        "original_label": LANG_MAP_INV.get(original_label, str(original_label)),
        "target_label": LANG_MAP_INV.get(target_label, str(target_label)),
        "tradeoff_curve": tradeoff,
    }

    if best_adv is not None:
        result["adversarial_audio"] = best_adv

    return result


# ---------------------------------------------------------------------------
# Load trained LID model
# ---------------------------------------------------------------------------

def load_lid_model(
    model_path: str,
    device: str = "cpu",
) -> FrameLevelLID:
    """Instantiate FrameLevelLID and load saved weights."""
    model = FrameLevelLID().to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    logger.info(f"Loaded LID model from {model_path}")
    return model


# ---------------------------------------------------------------------------
# Full attack pipeline
# ---------------------------------------------------------------------------

def run_attack(
    audio_path: str,
    lid_model_path: str,
    target_label_str: str = "en",
    output_dir: str = "outputs",
    device: str = None,
    sr: int = 22050,
) -> Dict:
    """End-to-end adversarial attack on a single audio file.

    1. Load audio.
    2. Load LID model.
    3. Determine original prediction (should be 'hi').
    4. Binary-search for minimum epsilon targeting *target_label*.
    5. Save adversarial audio and results JSON.
    """
    import soundfile as sf
    import librosa

    if device is None:
        from scripts.device_utils import get_device
        device = get_device()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- load audio ---------------------------------------------------
    audio_np, file_sr = sf.read(audio_path, dtype="float32")
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    if file_sr != sr:
        audio_np = librosa.resample(audio_np, orig_sr=file_sr, target_sr=sr)
    audio_tensor = torch.from_numpy(audio_np).float()

    # --- load model ---------------------------------------------------
    model = load_lid_model(lid_model_path, device)

    # --- original prediction ------------------------------------------
    original_pred = _predict_majority(model, audio_tensor, device, sr)
    logger.info(
        f"Original prediction: {LANG_MAP_INV.get(original_pred, original_pred)}"
    )

    target_label = LANG_MAP.get(target_label_str, 0)

    # --- binary search ------------------------------------------------
    logger.info(
        f"Running FGSM binary search: "
        f"{LANG_MAP_INV.get(original_pred)} -> {target_label_str}"
    )
    result = find_minimum_epsilon(
        model, audio_tensor, original_pred, target_label, device, sr,
    )

    # --- save adversarial audio ---------------------------------------
    adv_audio = result.pop("adversarial_audio", None)
    if adv_audio is not None:
        adv_path = out / "audio" / "adversarial_5s_segment.wav"
        adv_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(adv_path), adv_audio.numpy(), sr)
        result["adversarial_audio_path"] = str(adv_path)
        logger.info(f"Adversarial audio saved to {adv_path}")

    # --- save results JSON --------------------------------------------
    results_path = out / "adversarial_results.json"
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FGSM adversarial attack on LID model (Task 4.2)"
    )
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to input audio segment (≈5 s Hindi WAV)",
    )
    parser.add_argument(
        "--lid-model",
        type=str,
        default="models/lid/best_model.pt",
        help="Path to trained FrameLevelLID weights",
    )
    parser.add_argument(
        "--target-label",
        type=str,
        default="en",
        choices=list(LANG_MAP.keys()),
        help="Target language label to fool the model into predicting",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
    )
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    results = run_attack(
        audio_path=args.audio,
        lid_model_path=args.lid_model,
        target_label_str=args.target_label,
        output_dir=args.output_dir,
        device=args.device,
    )

    if results["flip_achieved"]:
        print(
            f"\nAttack SUCCEEDED: min epsilon={results['min_epsilon']:.6f}, "
            f"SNR={results['snr_at_min_db']:.1f} dB"
        )
        if results["snr_above_threshold"]:
            print("SNR > 40 dB — perturbation is imperceptible.")
        else:
            print("WARNING: SNR < 40 dB — perturbation may be audible.")
    else:
        print("\nAttack FAILED: could not flip prediction within search range.")
