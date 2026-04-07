"""
TTS data preparation for Speech Understanding PA-2.

Serves Tasks 3.1 (speaker embedding), 3.3 (prosody transfer), and 5 (voice cloning).

Processing steps:
  1. Process IndicVoices-R: quality filter, resample to 22050 Hz, loudness normalize
  2. Process Kathbath: extract usable segments, normalize
  3. Extract speaker embeddings via SpeechBrain ECAPA-TDNN / Resemblyzer
  4. Extract prosody features via Parselmouth: F0 contour, energy, duration

Output: data/processed/tts/
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "configs/dataset_config.yaml"
OUTPUT_DIR = Path("data/processed/tts")


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def quality_filter(
    audio_array: np.ndarray,
    sr: int,
    min_duration_sec: float = 2.0,
    max_duration_sec: float = 15.0,
    min_snr_db: float = 15.0,
) -> bool:
    """
    Apply quality filters to an audio clip.

    Checks duration bounds and estimated SNR.
    Returns True if audio passes all checks.
    """
    duration = len(audio_array) / sr
    if duration < min_duration_sec or duration > max_duration_sec:
        return False

    rms = np.sqrt(np.mean(audio_array ** 2))
    if rms < 1e-6:
        return False

    # Rough SNR estimation via signal/noise floor ratio
    sorted_power = np.sort(audio_array ** 2)
    noise_floor = np.mean(sorted_power[: len(sorted_power) // 10])
    signal_power = np.mean(sorted_power[len(sorted_power) // 2 :])
    if noise_floor > 0:
        snr = 10 * np.log10(signal_power / noise_floor)
        if snr < min_snr_db:
            return False

    return True


def resample_audio(
    audio_array: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """Resample audio to the target sample rate using librosa."""
    import librosa

    if orig_sr == target_sr:
        return audio_array
    return librosa.resample(audio_array, orig_sr=orig_sr, target_sr=target_sr)


def normalize_loudness(
    audio_array: np.ndarray,
    sr: int,
    target_lufs: float = -23.0,
) -> np.ndarray:
    """
    Normalize audio loudness to target LUFS using pyloudnorm.
    """
    import pyloudnorm as pyln

    meter = pyln.Meter(sr)
    current_lufs = meter.integrated_loudness(audio_array)
    if np.isinf(current_lufs):
        return audio_array
    return pyln.normalize.loudness(audio_array, current_lufs, target_lufs)


def process_indicvoices_r(
    data_dir: str,
    config: dict,
) -> List[Dict]:
    """
    Process IndicVoices-R dataset: quality filter, resample, normalize.

    Returns list of processed audio metadata dicts.
    """
    from datasets import load_from_disk

    audio_cfg = config["audio"]
    target_sr = audio_cfg["target_sample_rate"]
    target_lufs = audio_cfg["loudness_target_lufs"]

    split_path = Path(data_dir) / "train"
    if not split_path.exists():
        logger.warning(f"IndicVoices-R not found: {split_path}")
        return []

    ds = load_from_disk(str(split_path))

    if "audio" in ds.column_names:
        from datasets import Audio
        ds = ds.cast_column("audio", Audio(sampling_rate=48000))

    processed = []

    for idx, example in enumerate(ds):
        audio_data = example.get("audio")
        if audio_data is None:
            continue

        if isinstance(audio_data, dict):
            array = np.array(audio_data.get("array", []), dtype=np.float32)
            sr = audio_data.get("sampling_rate", 48000)
        else:
            decoded = audio_data
            if hasattr(decoded, "get"):
                array = np.array(decoded.get("array", []), dtype=np.float32)
                sr = decoded.get("sampling_rate", 48000)
            elif hasattr(decoded, "__getitem__"):
                array = np.array(decoded["array"], dtype=np.float32)
                sr = decoded["sampling_rate"]
            else:
                logger.warning(f"  Skipping idx {idx}: unknown audio format {type(audio_data)}")
                continue

        if not quality_filter(
            array, sr,
            min_duration_sec=audio_cfg["min_duration_sec"],
            max_duration_sec=audio_cfg["max_duration_sec"],
            min_snr_db=audio_cfg["min_snr_db"],
        ):
            continue

        array = resample_audio(array, sr, target_sr)
        array = normalize_loudness(array, target_sr, target_lufs)

        out_path = OUTPUT_DIR / "indicvoices_r" / f"{idx:06d}.npy"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out_path), array)

        processed.append({
            "id": f"ivr_{idx:06d}",
            "path": str(out_path),
            "duration_sec": len(array) / target_sr,
            "sr": target_sr,
            "speaker_id": example.get("speaker_id", "unknown"),
            "text": example.get("text", ""),
        })

    logger.info(f"Processed {len(processed)} / {len(ds)} IndicVoices-R clips")
    return processed


def process_kathbath(data_dir: str, config: dict) -> List[Dict]:
    """
    Process Kathbath dataset: extract usable segments, normalize.

    Returns list of processed audio metadata dicts.
    """
    from datasets import load_from_disk

    audio_cfg = config["audio"]
    target_sr = audio_cfg["target_sample_rate"]
    processed = []

    for split in ["train", "valid", "test"]:
        split_path = Path(data_dir) / split
        if not split_path.exists():
            continue
        ds = load_from_disk(str(split_path))
        if "audio" in ds.column_names:
            from datasets import Audio
            ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        for idx, example in enumerate(ds):
            audio_data = example.get("audio")
            if audio_data is None:
                continue

            if isinstance(audio_data, dict):
                array = np.array(audio_data.get("array", []), dtype=np.float32)
                sr = audio_data.get("sampling_rate", 16000)
            else:
                try:
                    array = np.array(audio_data["array"], dtype=np.float32)
                    sr = audio_data["sampling_rate"]
                except (KeyError, TypeError):
                    continue

            if not quality_filter(array, sr):
                continue

            array = resample_audio(array, sr, target_sr)
            array = normalize_loudness(array, target_sr)

            out_path = OUTPUT_DIR / "kathbath" / f"{split}_{idx:06d}.npy"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(out_path), array)

            processed.append({
                "id": f"kb_{split}_{idx:06d}",
                "path": str(out_path),
                "duration_sec": len(array) / target_sr,
                "sr": target_sr,
                "text": example.get("text", ""),
                "split": split,
            })

    logger.info(f"Processed {len(processed)} Kathbath clips")
    return processed


def extract_speaker_embeddings(
    audio_paths: List[str],
    method: str = "speechbrain",
) -> Dict[str, np.ndarray]:
    """
    Extract speaker embeddings from audio files.

    Args:
        audio_paths: Paths to processed audio numpy files.
        method: 'speechbrain' (ECAPA-TDNN) or 'resemblyzer'.

    Returns:
        Dict mapping audio_path -> embedding vector.
    """
    embeddings = {}

    if method == "speechbrain":
        from speechbrain.inference.speaker import EncoderClassifier
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"},
        )
        import torch
        for path in audio_paths:
            array = np.load(path)
            waveform = torch.tensor(array).unsqueeze(0)
            emb = classifier.encode_batch(waveform).squeeze().detach().numpy()
            embeddings[path] = emb

    elif method == "resemblyzer":
        from resemblyzer import VoiceEncoder, preprocess_wav
        encoder = VoiceEncoder()
        for path in audio_paths:
            array = np.load(path)
            wav = preprocess_wav(array)
            emb = encoder.embed_utterance(wav)
            embeddings[path] = emb

    logger.info(f"Extracted {len(embeddings)} speaker embeddings via {method}")
    return embeddings


def extract_prosody_features(
    audio_path: str,
    sr: int = 22050,
) -> Dict[str, np.ndarray]:
    """
    Extract prosody features using Parselmouth (Praat).

    Returns dict with:
      - f0: fundamental frequency contour
      - energy: RMS energy contour
      - duration_frames: number of analysis frames
    """
    import parselmouth
    from parselmouth.praat import call

    array = np.load(audio_path).astype(np.float64)
    snd = parselmouth.Sound(array, sampling_frequency=sr)

    pitch = call(snd, "To Pitch", 0.0, 75.0, 600.0)
    f0_values = pitch.selected_array["frequency"]
    f0_values[f0_values == 0] = np.nan

    intensity = call(snd, "To Intensity", 75.0, 0.0)
    energy_values = np.array(
        [call(intensity, "Get value in frame", i + 1)
         for i in range(call(intensity, "Get number of frames"))]
    )

    return {
        "f0": f0_values,
        "energy": energy_values,
        "duration_frames": len(f0_values),
    }


def save_manifest(records: List[Dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            serializable = {}
            for k, v in rec.items():
                if isinstance(v, np.ndarray):
                    serializable[k] = v.tolist()
                else:
                    serializable[k] = v
            f.write(json.dumps(serializable, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(records)} records to {output_path}")


def create_ljspeech_manifest(
    records: List[Dict],
    output_path: Path,
):
    """
    Write an LJSpeech-style pipe-delimited manifest.

    Format per line: <utterance_id>|<text>|<normalized_text>
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            utt_id = rec.get("id", "")
            text = rec.get("text", "").strip().replace("|", " ")
            normalized = text.lower()
            f.write(f"{utt_id}|{text}|{normalized}\n")
    logger.info(f"LJSpeech manifest: {len(records)} entries -> {output_path}")


def create_speaker_manifest(
    records: List[Dict],
    output_path: Path,
):
    """
    Create a speaker-level manifest with per-speaker statistics.

    Output JSON with: speaker_id, gender (if available), num_utterances,
    total_duration_sec, avg_duration_sec.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    speaker_stats: Dict[str, Dict] = {}

    for rec in records:
        spk = rec.get("speaker_id", "unknown")
        if spk not in speaker_stats:
            speaker_stats[spk] = {
                "speaker_id": spk,
                "gender": rec.get("gender", "unknown"),
                "num_utterances": 0,
                "total_duration_sec": 0.0,
                "utterance_ids": [],
            }
        speaker_stats[spk]["num_utterances"] += 1
        speaker_stats[spk]["total_duration_sec"] += rec.get("duration_sec", 0.0)
        speaker_stats[spk]["utterance_ids"].append(rec.get("id", ""))

    manifest = []
    for spk, stats in sorted(speaker_stats.items()):
        n = stats["num_utterances"]
        manifest.append({
            "speaker_id": stats["speaker_id"],
            "gender": stats["gender"],
            "num_utterances": n,
            "total_duration_sec": round(stats["total_duration_sec"], 2),
            "avg_duration_sec": round(stats["total_duration_sec"] / max(n, 1), 2),
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logger.info(
        f"Speaker manifest: {len(manifest)} speakers -> {output_path}"
    )


def prepare_tts_data(config_path: str):
    """Main entry point: prepare all TTS training data."""
    config = load_config(config_path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Step 1: Processing IndicVoices-R...")
    ivr_dir = config["datasets"]["indicvoices_r"]["save_dir"]
    ivr_records = process_indicvoices_r(ivr_dir, config)
    save_manifest(ivr_records, OUTPUT_DIR / "indicvoices_r_manifest.jsonl")

    logger.info("Step 2: Processing Kathbath...")
    kb_dir = config["datasets"]["kathbath"]["save_dir"]
    kb_records = process_kathbath(kb_dir, config)
    save_manifest(kb_records, OUTPUT_DIR / "kathbath_manifest.jsonl")

    all_records = ivr_records + kb_records
    all_paths = [r["path"] for r in all_records]

    logger.info("Step 3: Creating LJSpeech-style manifest...")
    create_ljspeech_manifest(all_records, OUTPUT_DIR / "ljspeech_manifest.txt")

    logger.info("Step 4: Creating speaker manifest...")
    create_speaker_manifest(all_records, OUTPUT_DIR / "speaker_manifest.json")

    logger.info("Step 5: Extracting speaker embeddings...")
    if all_paths:
        embeddings = extract_speaker_embeddings(all_paths[:100])
        emb_out = OUTPUT_DIR / "speaker_embeddings.npz"
        np.savez(str(emb_out), **{k: v for k, v in embeddings.items()})
        logger.info(f"Speaker embeddings saved to {emb_out}")

    logger.info("Step 6: Extracting prosody features...")
    prosody_records = []
    for path in all_paths[:50]:
        try:
            features = extract_prosody_features(path)
            prosody_records.append({"path": path, **features})
        except Exception as e:
            logger.warning(f"Prosody extraction failed for {path}: {e}")
    save_manifest(prosody_records, OUTPUT_DIR / "prosody_features.jsonl")

    logger.info("TTS data preparation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare TTS data for Tasks 3.1, 3.3, 5"
    )
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG,
        help="Path to dataset config YAML",
    )
    args = parser.parse_args()
    prepare_tts_data(args.config)
