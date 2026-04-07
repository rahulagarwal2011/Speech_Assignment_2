"""
Frame-level Language Identification (LID) model for Speech Understanding PA-2.

Task 1.1: Classify each 20ms frame as English (0), Hindi (1), or Maithili (2)
with a minimum F1-score of 0.85.

Architecture: CNN feature extractor -> Bidirectional GRU -> per-frame classifier
with temporal smoothing to prevent rapid flickering at language boundaries.

The model outputs per-frame language probabilities needed by Task 4.2 (FGSM).
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

LANG_MAP = {"en": 0, "hi": 1, "mai": 2}
LANG_MAP_INV = {v: k for k, v in LANG_MAP.items()}
NUM_CLASSES = 3
FRAME_RATE = 50  # 50 fps = 20ms per frame


class LIDDataset(Dataset):
    """Dataset for frame-level LID training.

    Each sample is a log-mel spectrogram tensor and corresponding
    per-frame language label array.
    """

    def __init__(self, manifest_path: str, max_frames: int = 250):
        self.records = []
        self.max_frames = max_frames
        path = Path(manifest_path)
        if path.exists():
            with open(path, "r") as f:
                for line in f:
                    rec = json.loads(line.strip())
                    self.records.append(rec)
        logger.info(f"LIDDataset loaded {len(self.records)} records from {manifest_path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        spec_path = rec.get("spectrogram_path", "")
        label_path = rec.get("label_path", "")

        if spec_path and Path(spec_path).exists():
            spec = torch.load(spec_path, weights_only=True)
        else:
            spec = torch.randn(80, self.max_frames)

        if label_path and Path(label_path).exists():
            labels = torch.load(label_path, weights_only=True)
        else:
            lang = rec.get("language", "en")
            lang_id = LANG_MAP.get(lang, 0)
            labels = torch.full((self.max_frames,), lang_id, dtype=torch.long)

        T = min(spec.shape[-1], labels.shape[0], self.max_frames)
        spec = spec[:, :T]
        labels = labels[:T]

        if T < self.max_frames:
            pad_len = self.max_frames - T
            spec = F.pad(spec, (0, pad_len))
            labels = F.pad(labels, (0, pad_len), value=-1)

        return spec, labels


class FrameLevelLID(nn.Module):
    """Frame-level Language Identification model.

    Architecture:
        1. CNN block: 3 conv layers extracting local spectral patterns
        2. BiGRU: captures temporal context across frames
        3. Frame classifier: per-frame 3-class prediction
        4. Temporal smoothing: median filter to stabilize predictions
    """

    def __init__(
        self,
        n_mels: int = 80,
        num_classes: int = NUM_CLASSES,
        cnn_channels: List[int] = None,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [32, 64, 128]

        layers = []
        in_ch = 1
        for out_ch in cnn_channels:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 1)),
            ])
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)

        cnn_freq_out = n_mels // (2 ** len(cnn_channels))
        rnn_input_size = cnn_channels[-1] * cnn_freq_out

        self.gru = nn.GRU(
            input_size=rnn_input_size,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden * 2, gru_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden, num_classes),
        )

        self.smoothing_kernel_size = 5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, n_mels, time) log-mel spectrogram

        Returns:
            (batch, time, num_classes) per-frame logits
        """
        x = x.unsqueeze(1)  # (B, 1, n_mels, T)
        x = self.cnn(x)     # (B, C, F', T)

        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * F)  # (B, T, C*F')

        x, _ = self.gru(x)     # (B, T, 2*H)
        logits = self.classifier(x)  # (B, T, num_classes)
        return logits

    def predict_smoothed(
        self, x: torch.Tensor, kernel_size: int = None
    ) -> torch.Tensor:
        """Predict with temporal median smoothing to reduce flickering."""
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)  # (B, T)

        if kernel_size is None:
            kernel_size = self.smoothing_kernel_size

        smoothed = []
        for b in range(preds.shape[0]):
            seq = preds[b].float()
            pad = kernel_size // 2
            padded = F.pad(seq.unsqueeze(0).unsqueeze(0), (pad, pad), mode="replicate")
            unfolded = padded.unfold(-1, kernel_size, 1).squeeze(0).squeeze(0)
            median_vals = unfolded.median(dim=-1).values
            smoothed.append(median_vals.long())

        return torch.stack(smoothed), probs

    def get_switch_timestamps(
        self, predictions: torch.Tensor, frame_rate: int = FRAME_RATE
    ) -> List[Dict]:
        """Extract language switch timestamps from frame predictions."""
        switches = []
        preds_np = predictions.cpu().numpy()
        for b in range(preds_np.shape[0]):
            seq = preds_np[b]
            for i in range(1, len(seq)):
                if seq[i] != seq[i - 1] and seq[i] >= 0 and seq[i - 1] >= 0:
                    switches.append({
                        "frame": i,
                        "time_sec": i / frame_rate,
                        "from_lang": LANG_MAP_INV.get(int(seq[i - 1]), "unk"),
                        "to_lang": LANG_MAP_INV.get(int(seq[i]), "unk"),
                    })
        return switches


def compute_lid_metrics(
    predictions: torch.Tensor, labels: torch.Tensor
) -> Dict[str, float]:
    """Compute per-class and overall F1 for LID evaluation."""
    mask = labels >= 0
    preds = predictions[mask].cpu().numpy()
    golds = labels[mask].cpu().numpy()

    metrics = {}
    for cls_id, cls_name in LANG_MAP_INV.items():
        tp = ((preds == cls_id) & (golds == cls_id)).sum()
        fp = ((preds == cls_id) & (golds != cls_id)).sum()
        fn = ((preds != cls_id) & (golds == cls_id)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[f"precision_{cls_name}"] = float(precision)
        metrics[f"recall_{cls_name}"] = float(recall)
        metrics[f"f1_{cls_name}"] = float(f1)

    correct = (preds == golds).sum()
    total = len(golds)
    metrics["accuracy"] = float(correct / total) if total > 0 else 0.0
    metrics["macro_f1"] = float(np.mean([metrics[f"f1_{n}"] for n in LANG_MAP_INV.values()]))
    return metrics


def train_lid_model(
    train_manifest: str,
    val_manifest: str,
    output_dir: str = "models/lid",
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = None,
):
    """Train the frame-level LID model."""
    if device is None:
        from scripts.device_utils import get_device
        device = get_device()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_ds = LIDDataset(train_manifest)
    val_ds = LIDDataset(val_manifest)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = FrameLevelLID().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    best_f1 = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for specs, labels in train_loader:
            specs = specs.to(device)
            labels = labels.to(device)

            logits = model(specs)  # (B, T, C)
            B, T, C = logits.shape
            loss = criterion(logits.reshape(-1, C), labels.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)

        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for specs, labels in val_loader:
                specs = specs.to(device)
                labels = labels.to(device)
                smoothed_preds, _ = model.predict_smoothed(specs)
                all_preds.append(smoothed_preds.cpu())
                all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        metrics = compute_lid_metrics(all_preds, all_labels)

        epoch_record = {"epoch": epoch, "loss": avg_loss, **metrics}
        history.append(epoch_record)
        logger.info(
            f"Epoch {epoch}/{epochs} | loss={avg_loss:.4f} | "
            f"macro_f1={metrics['macro_f1']:.4f} | acc={metrics['accuracy']:.4f}"
        )

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            torch.save(model.state_dict(), out_path / "best_model.pt")
            logger.info(f"  New best model saved (F1={best_f1:.4f})")

    torch.save(model.state_dict(), out_path / "final_model.pt")
    with open(out_path / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"Training complete. Best F1: {best_f1:.4f}")
    return best_f1, history


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Train frame-level LID model (Task 1.1)")
    parser.add_argument("--train", default="data/processed/lid/audio_train.jsonl")
    parser.add_argument("--val", default="data/processed/lid/audio_val.jsonl")
    parser.add_argument("--output", default="models/lid")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    train_lid_model(
        args.train, args.val, args.output,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, device=args.device,
    )
