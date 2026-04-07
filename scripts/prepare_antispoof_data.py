"""
Anti-spoofing data preparation for Speech Understanding PA-2.

Serves Task 6: adversarial robustness and spoofing detection (EER < 10%).

Processing steps:
  1. Create bonafide/spoof directory structure for ASVspoof-style evaluation
  2. Implement LFCC (Linear Frequency Cepstral Coefficients) feature extraction
  3. Generate placeholder manifest files for train/dev/eval protocols
  4. Prepare feature extraction pipeline for AASIST model training

Output: data/processed/antispoofing/
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "configs/dataset_config.yaml"
OUTPUT_DIR = Path("data/processed/antispoofing")


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_directory_structure():
    """
    Create bonafide/spoof directory structure mirroring ASVspoof conventions.
    """
    dirs = [
        OUTPUT_DIR / "bonafide" / "train",
        OUTPUT_DIR / "bonafide" / "dev",
        OUTPUT_DIR / "bonafide" / "eval",
        OUTPUT_DIR / "spoof" / "train",
        OUTPUT_DIR / "spoof" / "dev",
        OUTPUT_DIR / "spoof" / "eval",
        OUTPUT_DIR / "features" / "lfcc",
        OUTPUT_DIR / "features" / "spec",
        OUTPUT_DIR / "protocols",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created {len(dirs)} anti-spoofing directories")


def extract_lfcc(
    audio_array: np.ndarray,
    sr: int = 22050,
    n_lfcc: int = 60,
    n_fft: int = 1024,
    win_length_ms: float = 25.0,
    hop_length_ms: float = 10.0,
    use_torchaudio: bool = False,
) -> np.ndarray:
    """
    Extract Linear Frequency Cepstral Coefficients (LFCC).

    Two backends available:
      - scipy/librosa (default): manual linear filterbank + DCT
      - torchaudio: uses torchaudio.transforms.LFCC

    Args:
        audio_array: Raw waveform numpy array.
        sr: Sample rate.
        n_lfcc: Number of LFCC coefficients.
        n_fft: FFT size.
        win_length_ms: Window length in milliseconds.
        hop_length_ms: Hop length in milliseconds.
        use_torchaudio: If True, use torchaudio backend.

    Returns:
        LFCC feature matrix of shape (n_lfcc, time_frames).
    """
    win_length = int(win_length_ms * sr / 1000)
    hop_length = int(hop_length_ms * sr / 1000)

    if use_torchaudio:
        import torch
        import torchaudio

        waveform = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0)
        lfcc_transform = torchaudio.transforms.LFCC(
            sample_rate=sr,
            n_lfcc=n_lfcc,
            speckwargs={
                "n_fft": n_fft,
                "win_length": win_length,
                "hop_length": hop_length,
            },
        )
        lfcc = lfcc_transform(waveform).squeeze(0).numpy()
        return lfcc

    from scipy.fft import dct
    import librosa

    stft = np.abs(librosa.stft(
        audio_array,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
    )) ** 2

    n_filters = n_lfcc * 2
    freq_bins = stft.shape[0]
    linear_filterbank = np.zeros((n_filters, freq_bins))
    bin_edges = np.linspace(0, freq_bins - 1, n_filters + 2, dtype=int)

    for i in range(n_filters):
        start, center, end = bin_edges[i], bin_edges[i + 1], bin_edges[i + 2]
        if center > start:
            linear_filterbank[i, start:center] = np.linspace(0, 1, center - start)
        if end > center:
            linear_filterbank[i, center:end] = np.linspace(1, 0, end - center)

    filtered = np.dot(linear_filterbank, stft)
    log_filtered = np.log(filtered + 1e-8)
    lfcc = dct(log_filtered, type=2, axis=0, norm="ortho")[:n_lfcc]

    return lfcc


def extract_lfcc_delta(lfcc: np.ndarray, order: int = 2) -> np.ndarray:
    """
    Compute delta and double-delta features from LFCC.

    Returns concatenated [LFCC; delta-LFCC; delta-delta-LFCC].
    """
    import librosa

    features = [lfcc]
    for _ in range(order):
        delta = librosa.feature.delta(features[-1])
        features.append(delta)
    return np.concatenate(features, axis=0)


def extract_features_batch(
    audio_paths: List[str],
    output_dir: Path,
    sr: int = 22050,
    n_lfcc: int = 60,
    use_torchaudio: bool = False,
    include_deltas: bool = True,
) -> List[Dict]:
    """
    Batch-extract LFCC features (with optional deltas) for a list of audio files.

    Saves each feature matrix as a .npy file in output_dir and returns
    a manifest list with paths and metadata.

    Args:
        audio_paths: List of paths to audio files (.wav or .npy).
        output_dir: Directory to save extracted feature .npy files.
        sr: Sample rate.
        n_lfcc: Number of LFCC coefficients.
        use_torchaudio: Whether to use the torchaudio LFCC backend.
        include_deltas: Whether to append delta and delta-delta features.

    Returns:
        List of dicts with 'audio_path', 'feature_path', 'n_frames', 'n_coeffs'.
    """
    import soundfile as sf

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    n_success = 0
    n_fail = 0

    for idx, audio_path in enumerate(audio_paths):
        try:
            path = Path(audio_path)
            if path.suffix == ".npy":
                array = np.load(audio_path).astype(np.float32)
            elif path.suffix in (".wav", ".flac"):
                array, file_sr = sf.read(audio_path, dtype="float32")
                if file_sr != sr:
                    import librosa
                    array = librosa.resample(array, orig_sr=file_sr, target_sr=sr)
            else:
                logger.warning(f"Unsupported format: {path.suffix} ({audio_path})")
                n_fail += 1
                continue

            lfcc = extract_lfcc(array, sr=sr, n_lfcc=n_lfcc, use_torchaudio=use_torchaudio)

            if include_deltas:
                lfcc = extract_lfcc_delta(lfcc)

            feat_path = output_dir / f"lfcc_{idx:06d}.npy"
            np.save(str(feat_path), lfcc)

            manifest.append({
                "audio_path": str(audio_path),
                "feature_path": str(feat_path),
                "n_frames": lfcc.shape[1],
                "n_coeffs": lfcc.shape[0],
            })
            n_success += 1

        except Exception as e:
            logger.warning(f"Feature extraction failed for {audio_path}: {e}")
            n_fail += 1

        if (idx + 1) % 100 == 0:
            logger.info(f"  Processed {idx + 1}/{len(audio_paths)} files")

    logger.info(
        f"Batch feature extraction: {n_success} succeeded, {n_fail} failed "
        f"out of {len(audio_paths)} files"
    )
    return manifest


def create_protocol_manifest(
    bonafide_dir: Path,
    spoof_dir: Path,
    split: str,
    output_path: Path,
):
    """
    Create an ASVspoof-style protocol file listing all trials.

    Format: SPEAKER_ID UTTERANCE_ID - SYSTEM_ID KEY
    where KEY is 'bonafide' or 'spoof'.
    """
    entries = []

    bonafide_split = bonafide_dir / split
    if bonafide_split.exists():
        for f in sorted(bonafide_split.iterdir()):
            if f.suffix in (".wav", ".npy", ".flac"):
                entries.append(f"- {f.stem} - - bonafide")

    spoof_split = spoof_dir / split
    if spoof_split.exists():
        for f in sorted(spoof_split.iterdir()):
            if f.suffix in (".wav", ".npy", ".flac"):
                entries.append(f"- {f.stem} - A00 spoof")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(entries) + "\n")
    logger.info(f"Protocol manifest ({split}): {len(entries)} entries -> {output_path}")


def process_bonafide_from_tts(tts_manifest_path: Path):
    """
    Copy processed TTS audio as bonafide samples for anti-spoofing training.

    Reads the TTS manifest and symlinks/copies audio into bonafide dirs.
    """
    if not tts_manifest_path.exists():
        logger.warning(f"TTS manifest not found: {tts_manifest_path}")
        return

    records = []
    with open(tts_manifest_path, "r") as f:
        for line in f:
            records.append(json.loads(line))

    n_train = int(len(records) * 0.8)
    n_dev = int(len(records) * 0.1)

    splits = {
        "train": records[:n_train],
        "dev": records[n_train : n_train + n_dev],
        "eval": records[n_train + n_dev :],
    }

    for split, recs in splits.items():
        out_dir = OUTPUT_DIR / "bonafide" / split
        out_dir.mkdir(parents=True, exist_ok=True)
        for rec in recs:
            src = Path(rec.get("path", ""))
            if src.exists():
                dst = out_dir / src.name
                if not dst.exists():
                    os.symlink(src.resolve(), dst)

    logger.info(
        f"Linked bonafide audio: train={len(splits['train'])}, "
        f"dev={len(splits['dev'])}, eval={len(splits['eval'])}"
    )


def prepare_antispoof_data(config_path: str):
    """Main entry point: prepare all anti-spoofing data."""
    config = load_config(config_path)

    logger.info("Step 1: Creating directory structure...")
    create_directory_structure()

    logger.info("Step 2: Linking bonafide samples from TTS data...")
    tts_manifest = Path("data/processed/tts/indicvoices_r_manifest.jsonl")
    process_bonafide_from_tts(tts_manifest)

    logger.info("Step 3: Creating protocol manifests...")
    for split in ["train", "dev", "eval"]:
        create_protocol_manifest(
            OUTPUT_DIR / "bonafide",
            OUTPUT_DIR / "spoof",
            split,
            OUTPUT_DIR / "protocols" / f"ASVspoof2024.LA.cm.{split}.trl.txt",
        )

    logger.info("Step 4: Batch-extracting LFCC features for bonafide audio...")
    bonafide_paths = []
    for split in ["train", "dev", "eval"]:
        split_dir = OUTPUT_DIR / "bonafide" / split
        if split_dir.exists():
            for f in sorted(split_dir.iterdir()):
                if f.suffix in (".wav", ".npy", ".flac"):
                    bonafide_paths.append(str(f))

    if bonafide_paths:
        feat_manifest = extract_features_batch(
            bonafide_paths,
            OUTPUT_DIR / "features" / "lfcc",
        )
        feat_manifest_path = OUTPUT_DIR / "features" / "lfcc_manifest.jsonl"
        feat_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(feat_manifest_path, "w") as f:
            for rec in feat_manifest:
                f.write(json.dumps(rec) + "\n")
        logger.info(f"LFCC feature manifest: {len(feat_manifest)} -> {feat_manifest_path}")
    else:
        logger.info("No bonafide audio found yet; skipping batch feature extraction.")

    logger.info(
        "Anti-spoofing data preparation complete. "
        "Spoof samples will be generated by the voice cloning pipeline."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare anti-spoofing data for Task 6"
    )
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG,
        help="Path to dataset config YAML",
    )
    args = parser.parse_args()
    prepare_antispoof_data(args.config)
