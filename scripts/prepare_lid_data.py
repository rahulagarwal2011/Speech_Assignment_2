"""
LID (Language Identification) data preparation for Speech Understanding PA-2.

Serves Tasks 1.1 (text LID) and 3.2 (audio LID / code-switch detection).

Processing steps:
  1. Load Kathbath audio splits for Maithili speech segments
  2. Load Bhasha-Abhijnaanam text for Hindi/Maithili/English text discrimination
  3. Segment audio into 5-second chunks with overlap
  4. Create frame-level language labels at 50 fps
  5. Compute 80-dim log-mel spectrograms (25ms window, 10ms hop)
  6. Train/val/test split (80/10/10)
  7. Save processed features as JSONL manifests

Output: data/processed/lid/
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "configs/dataset_config.yaml"
OUTPUT_DIR = Path("data/processed/lid")


def load_config(config_path: str) -> dict:
    """Load dataset configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_kathbath_audio(data_dir: str) -> List[dict]:
    """
    Load Kathbath Maithili audio dataset from disk.

    Returns list of dicts with keys: audio_path, text, duration, speaker_id.
    """
    from datasets import load_from_disk

    records = []
    for split in ["train", "valid", "test"]:
        split_path = Path(data_dir) / split
        if not split_path.exists():
            logger.warning(f"Kathbath split not found: {split_path}")
            continue
        ds = load_from_disk(str(split_path))
        for example in ds:
            records.append({
                "audio": example.get("audio"),
                "text": example.get("text", ""),
                "split": split,
            })
    logger.info(f"Loaded {len(records)} Kathbath examples")
    return records


def load_bhasha_abhijnaanam_text(data_dir: str) -> List[dict]:
    """
    Load Bhasha-Abhijnaanam text LID dataset.

    Returns list of dicts with keys: text, language, script.
    """
    from datasets import load_from_disk

    records = []
    split_path = Path(data_dir) / "train"
    if not split_path.exists():
        logger.warning(f"Bhasha-Abhijnaanam data not found: {split_path}")
        return records
    ds = load_from_disk(str(split_path))
    for example in ds:
        records.append({
            "text": example.get("text", ""),
            "language": example.get("language", ""),
            "script": example.get("script", ""),
        })
    logger.info(f"Loaded {len(records)} Bhasha-Abhijnaanam text examples")
    return records


def segment_audio_chunks(
    audio_array: np.ndarray,
    sr: int,
    chunk_duration_sec: float = 5.0,
    hop_duration_sec: float = 2.5,
) -> List[np.ndarray]:
    """
    Segment audio into fixed-length chunks with overlap.

    Args:
        audio_array: Raw waveform as numpy array.
        sr: Sample rate.
        chunk_duration_sec: Duration of each chunk in seconds.
        hop_duration_sec: Hop between chunk starts in seconds.

    Returns:
        List of audio chunk arrays.
    """
    chunk_samples = int(chunk_duration_sec * sr)
    hop_samples = int(hop_duration_sec * sr)
    chunks = []
    for start in range(0, len(audio_array) - chunk_samples + 1, hop_samples):
        chunks.append(audio_array[start : start + chunk_samples])
    return chunks


def create_frame_level_labels(
    language_segments: List[Dict],
    total_duration_sec: float,
    fps: int = 50,
) -> np.ndarray:
    """
    Create frame-level language labels at the given FPS.

    Args:
        language_segments: List of dicts with 'start', 'end', 'language' keys.
        total_duration_sec: Total duration of the utterance in seconds.
        fps: Frames per second for the label resolution.

    Returns:
        Integer array of shape (num_frames,) with language IDs per frame.
    """
    lang_map = {"english": 0, "hindi": 1, "maithili": 2, "silence": 3}
    num_frames = int(total_duration_sec * fps)
    labels = np.full(num_frames, lang_map["silence"], dtype=np.int32)

    for seg in language_segments:
        start_frame = int(seg["start"] * fps)
        end_frame = int(seg["end"] * fps)
        lang_id = lang_map.get(seg["language"], 3)
        labels[start_frame:end_frame] = lang_id

    return labels


def compute_log_mel_spectrogram(
    audio_array: np.ndarray,
    sr: int,
    n_mels: int = 80,
    n_fft: int = 1024,
    win_length_ms: float = 25.0,
    hop_length_ms: float = 10.0,
) -> np.ndarray:
    """
    Compute 80-dim log-mel spectrogram.

    Args:
        audio_array: Raw waveform numpy array.
        sr: Sample rate.
        n_mels: Number of mel filterbank channels.
        n_fft: FFT size.
        win_length_ms: Window length in milliseconds.
        hop_length_ms: Hop length in milliseconds.

    Returns:
        Log-mel spectrogram of shape (n_mels, time_frames).
    """
    import librosa

    win_length = int(win_length_ms * sr / 1000)
    hop_length = int(hop_length_ms * sr / 1000)

    mel_spec = librosa.feature.melspectrogram(
        y=audio_array,
        sr=sr,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel


def split_data(
    records: List[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Split records into train/val/test sets."""
    np.random.seed(42)
    indices = np.random.permutation(len(records))
    n_train = int(len(records) * train_ratio)
    n_val = int(len(records) * val_ratio)

    train = [records[i] for i in indices[:n_train]]
    val = [records[i] for i in indices[n_train : n_train + n_val]]
    test = [records[i] for i in indices[n_train + n_val :]]
    return train, val, test


def save_jsonl(records: List[dict], output_path: Path):
    """Write records as newline-delimited JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(records)} records to {output_path}")


def prepare_audio_lid_data(
    audio_records: List[dict],
    output_dir: Path,
    sr: int = 16000,
    chunk_duration_sec: float = 5.0,
    hop_duration_sec: float = 2.5,
    n_mels: int = 80,
    fps: int = 50,
) -> List[dict]:
    """
    Process Kathbath audio into spectrograms and frame-level labels saved as .pt tensors.

    For each audio record, segments into chunks, computes log-mel spectrograms, creates
    frame-level labels, and saves both as .pt files. Returns manifest records with
    spectrogram_path and label_path fields compatible with train_lid.py's LIDDataset.

    Args:
        audio_records: Raw records from load_kathbath_audio().
        output_dir: Base directory for saving .pt files.
        sr: Expected sample rate from Kathbath.
        chunk_duration_sec: Chunk length in seconds.
        hop_duration_sec: Hop between chunks in seconds.
        n_mels: Number of mel bands.
        fps: Frame rate for labels (50 fps = 20ms resolution).

    Returns:
        List of manifest dicts with spectrogram_path, label_path, and metadata.
    """
    spec_dir = output_dir / "spectrograms"
    label_dir = output_dir / "labels"
    spec_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    manifest_records = []
    chunk_idx = 0

    for rec_idx, rec in enumerate(audio_records):
        audio_data = rec.get("audio")
        if audio_data is None:
            continue

        if isinstance(audio_data, dict):
            array = np.array(audio_data.get("array", []), dtype=np.float32)
            rec_sr = audio_data.get("sampling_rate", sr)
        elif isinstance(audio_data, np.ndarray):
            array = audio_data.astype(np.float32)
            rec_sr = sr
        else:
            continue

        if len(array) == 0:
            continue

        if rec_sr != sr:
            import librosa
            array = librosa.resample(array, orig_sr=rec_sr, target_sr=sr)

        chunks = segment_audio_chunks(array, sr, chunk_duration_sec, hop_duration_sec)

        language_label = "maithili"
        for ci, chunk in enumerate(chunks):
            chunk_duration = len(chunk) / sr

            spec = compute_log_mel_spectrogram(chunk, sr, n_mels=n_mels)
            spec_tensor = torch.tensor(spec, dtype=torch.float32)

            n_spec_frames = spec.shape[1]
            language_segments = [{"start": 0.0, "end": chunk_duration, "language": language_label}]
            frame_labels = create_frame_level_labels(language_segments, chunk_duration, fps=fps)
            n_label_frames = min(len(frame_labels), n_spec_frames)
            frame_labels = frame_labels[:n_label_frames]
            label_tensor = torch.tensor(frame_labels, dtype=torch.long)

            if n_spec_frames > n_label_frames:
                spec_tensor = spec_tensor[:, :n_label_frames]

            spec_path = spec_dir / f"spec_{chunk_idx:07d}.pt"
            label_path = label_dir / f"label_{chunk_idx:07d}.pt"
            torch.save(spec_tensor, spec_path)
            torch.save(label_tensor, label_path)

            manifest_records.append({
                "spectrogram_path": str(spec_path),
                "label_path": str(label_path),
                "language": language_label,
                "duration_sec": round(chunk_duration, 3),
                "n_frames": int(n_label_frames),
                "source": "kathbath",
                "split": rec.get("split", "train"),
                "source_record_idx": rec_idx,
                "chunk_idx": ci,
            })
            chunk_idx += 1

        if (rec_idx + 1) % 200 == 0:
            logger.info(f"  Processed {rec_idx + 1}/{len(audio_records)} audio records ({chunk_idx} chunks)")

    logger.info(f"Audio LID data: {chunk_idx} chunks from {len(audio_records)} records")
    return manifest_records


def prepare_lid_data(config_path: str):
    """Main entry point: prepare all LID training data."""
    config = load_config(config_path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    kathbath_dir = config["datasets"]["kathbath"]["save_dir"]
    bhasha_dir = config["datasets"]["bhasha_abhijnaanam"]["save_dir"]

    logger.info("Loading Kathbath audio...")
    audio_records = load_kathbath_audio(kathbath_dir)

    logger.info("Processing audio into spectrograms and frame labels...")
    spec_records = prepare_audio_lid_data(audio_records, OUTPUT_DIR)

    logger.info("Splitting audio spectrogram data...")
    train_spec, val_spec, test_spec = split_data(spec_records)
    save_jsonl(train_spec, OUTPUT_DIR / "audio_train.jsonl")
    save_jsonl(val_spec, OUTPUT_DIR / "audio_val.jsonl")
    save_jsonl(test_spec, OUTPUT_DIR / "audio_test.jsonl")

    logger.info("Loading Bhasha-Abhijnaanam text LID data...")
    text_records = load_bhasha_abhijnaanam_text(bhasha_dir)

    logger.info("Splitting text LID data...")
    train_t, val_t, test_t = split_data(text_records)
    save_jsonl(train_t, OUTPUT_DIR / "text_train.jsonl")
    save_jsonl(val_t, OUTPUT_DIR / "text_val.jsonl")
    save_jsonl(test_t, OUTPUT_DIR / "text_test.jsonl")

    logger.info("LID data preparation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare LID data for Tasks 1.1 and 3.2"
    )
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG,
        help="Path to dataset config YAML",
    )
    args = parser.parse_args()
    prepare_lid_data(args.config)
