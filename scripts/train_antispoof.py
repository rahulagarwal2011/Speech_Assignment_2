"""
LFCC-based anti-spoofing classifier for Speech Understanding PA-2 Task 4.1.

Trains a Light CNN (LCNN) binary classifier to distinguish bonafide speech
from voice-cloned (spoof) speech using LFCC + delta + delta-delta features.
Target: EER < 10%.

Expects:
  - Bonafide audio: student voice recording (e.g. outputs/audio/student_voice_ref.wav)
  - Spoof audio: synthesized lecture (e.g. outputs/audio/output_LRL_cloned.wav)

Usage:
  python scripts/train_antispoof.py \
      --bonafide-audio outputs/audio/student_voice_ref.wav \
      --spoof-audio outputs/audio/output_LRL_cloned.wav \
      --output-dir models/antispoof \
      --epochs 50
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Audio segmentation
# ---------------------------------------------------------------------------

def segment_audio(
    audio_path: str,
    segment_sec: float = 4.0,
    hop_sec: float = 2.0,
    sr: int = 22050,
) -> List[np.ndarray]:
    """Segment a WAV file into fixed-length overlapping clips.

    Args:
        audio_path: Path to the WAV file.
        segment_sec: Length of each segment in seconds.
        hop_sec: Hop between consecutive segments in seconds.
        sr: Target sample rate.

    Returns:
        List of 1-D numpy arrays, each containing one segment.
    """
    import soundfile as sf
    import librosa

    audio, file_sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if file_sr != sr:
        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)

    seg_samples = int(segment_sec * sr)
    hop_samples = int(hop_sec * sr)

    segments: List[np.ndarray] = []
    start = 0
    while start + seg_samples <= len(audio):
        segments.append(audio[start : start + seg_samples].copy())
        start += hop_samples

    if not segments and len(audio) > sr:
        padded = np.zeros(seg_samples, dtype=np.float32)
        padded[: len(audio)] = audio
        segments.append(padded)

    logger.info(
        f"Segmented {audio_path}: {len(segments)} clips "
        f"({segment_sec}s window, {hop_sec}s hop)"
    )
    return segments


# ---------------------------------------------------------------------------
# Feature extraction (re-uses prepare_antispoof_data utilities)
# ---------------------------------------------------------------------------

def extract_lfcc_with_deltas(
    audio_array: np.ndarray,
    sr: int = 22050,
    n_lfcc: int = 20,
    n_fft: int = 512,
    win_length_ms: float = 20.0,
    hop_length_ms: float = 10.0,
) -> np.ndarray:
    """Extract LFCC features with delta and delta-delta appended.

    Returns array of shape (60, T) = 20 static + 20 delta + 20 delta-delta.
    """
    from scripts.prepare_antispoof_data import extract_lfcc, extract_lfcc_delta

    lfcc = extract_lfcc(
        audio_array,
        sr=sr,
        n_lfcc=n_lfcc,
        n_fft=n_fft,
        win_length_ms=win_length_ms,
        hop_length_ms=hop_length_ms,
    )
    return extract_lfcc_delta(lfcc, order=2)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AntiSpoofDataset(Dataset):
    """Binary anti-spoofing dataset: bonafide (1) vs spoof (0).

    Loads pre-segmented WAV clips from two directories, extracts LFCC+delta
    features on the fly, and pads/truncates to a fixed number of frames.
    """

    def __init__(
        self,
        bonafide_clips: List[np.ndarray],
        spoof_clips: List[np.ndarray],
        sr: int = 22050,
        n_lfcc: int = 20,
        max_frames: int = 400,
    ):
        super().__init__()
        self.sr = sr
        self.n_lfcc = n_lfcc
        self.max_frames = max_frames

        self.items: List[Tuple[np.ndarray, int]] = []
        for clip in bonafide_clips:
            self.items.append((clip, 1))
        for clip in spoof_clips:
            self.items.append((clip, 0))

        logger.info(
            f"AntiSpoofDataset: {len(bonafide_clips)} bonafide, "
            f"{len(spoof_clips)} spoof, {len(self.items)} total"
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        audio, label = self.items[idx]

        feat = extract_lfcc_with_deltas(
            audio, sr=self.sr, n_lfcc=self.n_lfcc,
        )  # (60, T)

        T = feat.shape[1]
        if T > self.max_frames:
            feat = feat[:, : self.max_frames]
        elif T < self.max_frames:
            pad = np.zeros((feat.shape[0], self.max_frames - T), dtype=np.float32)
            feat = np.concatenate([feat, pad], axis=1)

        return torch.from_numpy(feat.astype(np.float32)), label


# ---------------------------------------------------------------------------
# LCNN model with Max-Feature-Map activation
# ---------------------------------------------------------------------------

class MaxFeatureMap(nn.Module):
    """Max-Feature-Map (MFM) activation: splits channels in half and takes
    element-wise max, halving the channel dimension."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        return torch.max(x[:, : c // 2], x[:, c // 2 :])


class LCNNAntiSpoof(nn.Module):
    """Light CNN for binary anti-spoofing classification.

    Architecture follows the LCNN design from ASVspoof baselines:
      Conv+MFM blocks -> adaptive pooling -> FC layers -> single logit.
    """

    def __init__(self, in_channels: int = 60):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
            MaxFeatureMap(),                           # -> 32 ch
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            MaxFeatureMap(),                           # -> 32 ch
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            MaxFeatureMap(),                           # -> 64 ch
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            MaxFeatureMap(),                           # -> 32 ch
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, 60, T) LFCC+delta features.

        Returns:
            (batch,) raw logits (apply sigmoid for probability).
        """
        x = x.unsqueeze(1)       # (B, 1, 60, T)
        x = self.features(x)     # (B, 32, F', T')
        x = self.pool(x)         # (B, 32, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 32)
        return self.classifier(x).squeeze(-1)  # (B,)


# ---------------------------------------------------------------------------
# EER computation
# ---------------------------------------------------------------------------

def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute Equal Error Rate.

    Args:
        y_true: Binary ground-truth labels (1 = bonafide, 0 = spoof).
        y_scores: Predicted scores (higher = more likely bonafide).

    Returns:
        EER as a float in [0, 1].
    """
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return float(eer)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_antispoof(
    bonafide_audio: str,
    spoof_audio: str,
    output_dir: str = "models/antispoof",
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-3,
    segment_sec: float = 4.0,
    hop_sec: float = 2.0,
    device: str = None,
) -> Dict:
    """Full training pipeline for the LCNN anti-spoofing model.

    1. Segments bonafide and spoof audio into clips.
    2. Splits into train / val / test (70/15/15).
    3. Trains LCNN with BCE loss.
    4. Evaluates EER on the test set.
    5. Saves best model and results.
    """
    if device is None:
        from scripts.device_utils import get_device
        device = get_device()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- segment audio ------------------------------------------------
    bonafide_clips = segment_audio(bonafide_audio, segment_sec, hop_sec)
    spoof_clips = segment_audio(spoof_audio, segment_sec, hop_sec)

    if not bonafide_clips or not spoof_clips:
        raise RuntimeError(
            "Segmentation produced no clips. Check that the audio files "
            "exist and are long enough."
        )

    # --- balance classes ----------------------------------------------
    min_clips = min(len(bonafide_clips), len(spoof_clips))
    rng = np.random.RandomState(42)
    rng.shuffle(bonafide_clips)
    rng.shuffle(spoof_clips)
    bonafide_clips = bonafide_clips[:min_clips]
    spoof_clips = spoof_clips[:min_clips]
    logger.info(f"Balanced: {min_clips} clips per class")

    # --- train / val / test split (70/15/15) --------------------------
    n = min_clips
    n_train = max(1, int(0.7 * n))
    n_val = max(1, int(0.15 * n))

    def _split(clips):
        return clips[:n_train], clips[n_train : n_train + n_val], clips[n_train + n_val :]

    bf_train, bf_val, bf_test = _split(bonafide_clips)
    sp_train, sp_val, sp_test = _split(spoof_clips)

    train_ds = AntiSpoofDataset(bf_train, sp_train)
    val_ds = AntiSpoofDataset(bf_val, sp_val)
    test_ds = AntiSpoofDataset(bf_test, sp_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # --- model / optimiser -------------------------------------------
    model = LCNNAntiSpoof(in_channels=60).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_eer = 1.0
    history: List[Dict] = []

    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        running_loss = 0.0
        n_batches = 0
        for feats, labels in train_loader:
            feats = feats.to(device)
            labels = labels.float().to(device)

            logits = model(feats)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = running_loss / max(n_batches, 1)

        # --- validate ---
        val_eer = _evaluate_eer(model, val_loader, device)

        record = {"epoch": epoch, "loss": avg_loss, "val_eer": val_eer}
        history.append(record)
        logger.info(
            f"Epoch {epoch}/{epochs} | loss={avg_loss:.4f} | val_EER={val_eer:.4f}"
        )

        if val_eer < best_eer:
            best_eer = val_eer
            torch.save(model.state_dict(), out / "best_antispoof.pt")
            logger.info(f"  -> New best model (EER={best_eer:.4f})")

    # --- test ---------------------------------------------------------
    model.load_state_dict(torch.load(out / "best_antispoof.pt", weights_only=True))
    test_eer = _evaluate_eer(model, test_loader, device)
    logger.info(f"Test EER: {test_eer:.4f} (target < 0.10)")

    torch.save(model.state_dict(), out / "final_antispoof.pt")

    results = {
        "best_val_eer": best_eer,
        "test_eer": test_eer,
        "epochs": epochs,
        "bonafide_clips": len(bonafide_clips),
        "spoof_clips": len(spoof_clips),
        "history": history,
    }
    with open(out / "antispoof_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def _evaluate_eer(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> float:
    """Run inference on a DataLoader and compute EER."""
    model.eval()
    all_labels: List[int] = []
    all_scores: List[float] = []

    with torch.no_grad():
        for feats, labels in loader:
            feats = feats.to(device)
            logits = model(feats)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_scores.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())

    if len(set(all_labels)) < 2:
        logger.warning("Only one class present in evaluation set; returning EER=0.5")
        return 0.5

    return compute_eer(np.array(all_labels), np.array(all_scores))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train LCNN anti-spoofing classifier (Task 4.1)"
    )
    parser.add_argument(
        "--bonafide-audio",
        type=str,
        default="outputs/audio/student_voice_ref.wav",
        help="Path to bonafide (real) audio file",
    )
    parser.add_argument(
        "--spoof-audio",
        type=str,
        default="outputs/audio/output_LRL_cloned.wav",
        help="Path to spoof (synthesized) audio file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/antispoof",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--segment-sec", type=float, default=4.0)
    parser.add_argument("--hop-sec", type=float, default=2.0)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    results = train_antispoof(
        bonafide_audio=args.bonafide_audio,
        spoof_audio=args.spoof_audio,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        segment_sec=args.segment_sec,
        hop_sec=args.hop_sec,
        device=args.device,
    )

    print(f"\nTest EER: {results['test_eer']:.4f}")
    target = 0.10
    if results["test_eer"] < target:
        print(f"PASS: EER < {target}")
    else:
        print(f"NEEDS IMPROVEMENT: EER >= {target}")
