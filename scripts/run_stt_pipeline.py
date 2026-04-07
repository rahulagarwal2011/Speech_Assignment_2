"""
Full Part I pipeline: Denoise -> LID -> Constrained Whisper Decoding.

Orchestrates the three stages of the speech-to-text pipeline for
code-switched Maithili/Hindi/English lecture audio:

  1. Denoise the raw lecture audio (DeepFilterNet or spectral subtraction)
  2. Run frame-level LID to identify language boundaries
  3. Run constrained Whisper decoding with language-aware segmentation
  4. Merge LID labels and transcript into a unified output JSON
  5. Optionally compute WER against ground truth
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load pipeline configuration from YAML."""
    path = Path(config_path)
    if not path.exists():
        logger.warning(f"Config not found at {config_path}; using defaults")
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f)


def stage_denoise(
    audio_path: str,
    output_dir: str,
    method: str = "deepfilternet",
) -> str:
    """Stage 1: Denoise the input audio."""
    from scripts.denoise_audio import denoise

    out_path = Path(output_dir) / "audio"
    out_path.mkdir(parents=True, exist_ok=True)
    denoised_path = str(out_path / "denoised_segment.wav")

    logger.info(f"[Stage 1] Denoising: {audio_path} -> {denoised_path} (method={method})")
    denoise(audio_path, denoised_path, method=method)
    return denoised_path


def stage_lid(
    audio_path: str,
    model_path: str = "models/lid/best_model.pt",
    device: Optional[str] = None,
) -> List[Dict]:
    """Stage 2: Run frame-level LID to extract language segments.

    Loads the trained FrameLevelLID model, computes log-mel spectrogram
    for the denoised audio, runs frame-level prediction with temporal
    smoothing, and returns contiguous language segments with timestamps.
    """
    import torchaudio

    from scripts.train_lid import FrameLevelLID, FRAME_RATE, LANG_MAP_INV

    if device is None:
        from scripts.device_utils import get_device
        device = get_device()

    model = FrameLevelLID()
    model_file = Path(model_path)
    lid_trained = False
    if model_file.exists():
        state_dict = torch.load(str(model_file), map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded LID model from {model_path}")
        lid_trained = True
    else:
        logger.warning(f"LID model not found at {model_path}; skipping LID segmentation")
        return []

    model = model.to(device)
    model.eval()

    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    target_sr = 16000
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
        sr = target_sr

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_mels=80, n_fft=400, hop_length=int(sr / FRAME_RATE),
    )
    mel_spec = mel_transform(waveform)
    log_mel = torch.log(mel_spec + 1e-8)  # (1, 80, T)

    with torch.no_grad():
        log_mel_dev = log_mel.to(device)
        smoothed_preds, probs = model.predict_smoothed(log_mel_dev)

    preds_np = smoothed_preds.cpu().numpy()[0]
    probs_np = probs.cpu().numpy()[0]

    segments = _merge_frame_predictions(preds_np, probs_np, FRAME_RATE, LANG_MAP_INV)
    logger.info(f"[Stage 2] LID found {len(segments)} language segments")

    return segments


def _merge_frame_predictions(
    preds: np.ndarray,
    probs: np.ndarray,
    frame_rate: int,
    lang_map_inv: Dict[int, str],
    min_segment_frames: int = 25,
) -> List[Dict]:
    """Merge contiguous same-language frames into segments.

    Short segments (< min_segment_frames) are absorbed into the
    surrounding dominant language to avoid micro-switches.
    """
    if len(preds) == 0:
        return []

    raw_segments = []
    current_lang = int(preds[0])
    start_frame = 0

    for i in range(1, len(preds)):
        if int(preds[i]) != current_lang:
            raw_segments.append({
                "language": lang_map_inv.get(current_lang, "unk"),
                "lang_id": current_lang,
                "start_frame": start_frame,
                "end_frame": i,
                "avg_confidence": float(np.mean(np.max(probs[start_frame:i], axis=-1))),
            })
            current_lang = int(preds[i])
            start_frame = i

    raw_segments.append({
        "language": lang_map_inv.get(current_lang, "unk"),
        "lang_id": current_lang,
        "start_frame": start_frame,
        "end_frame": len(preds),
        "avg_confidence": float(np.mean(np.max(probs[start_frame:], axis=-1))),
    })

    merged = []
    for seg in raw_segments:
        duration_frames = seg["end_frame"] - seg["start_frame"]
        if duration_frames < min_segment_frames and merged:
            merged[-1]["end_frame"] = seg["end_frame"]
        else:
            merged.append(seg)

    for seg in merged:
        seg["start_time"] = seg["start_frame"] / frame_rate
        seg["end_time"] = seg["end_frame"] / frame_rate

    return merged


def stage_transcribe(
    audio_path: str,
    language_segments: List[Dict],
    ngram_lm_path: str = "data/processed/ngram/maithili_3gram.arpa",
    technical_vocab_path: str = "data/processed/ngram/technical_vocab.json",
    model_name: str = "large-v3",
    alpha: float = 0.3,
    beta: float = 3.0,
    device: Optional[str] = None,
) -> List[Dict]:
    """Stage 3: Run constrained Whisper decoding with language-aware segments."""
    from scripts.constrained_decode import (
        NgramLogitBiasProcessor,
        WhisperConstrainedTranscriber,
    )

    tokenizer = None
    try:
        import whisper
        tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True)
    except Exception:
        logger.warning("Could not load Whisper tokenizer; n-gram biasing may be limited")

    ngram_processor = NgramLogitBiasProcessor(
        ngram_lm_path=ngram_lm_path,
        technical_vocab_path=technical_vocab_path,
        tokenizer=tokenizer,
        alpha=alpha,
        beta=beta,
    )

    transcriber = WhisperConstrainedTranscriber(
        model_name=model_name,
        ngram_processor=ngram_processor,
        device=device,
    )

    segments = transcriber.transcribe(audio_path, language_segments=language_segments)
    logger.info(f"[Stage 3] Transcribed {len(segments)} segments")
    return segments


def merge_lid_and_transcript(
    lid_segments: List[Dict],
    transcript_segments: List[Dict],
) -> List[Dict]:
    """Merge LID language labels into transcript segments.

    For each transcript segment, find the overlapping LID segment(s)
    and tag with the dominant language.
    """
    for t_seg in transcript_segments:
        t_start = t_seg.get("start_time", 0)
        t_end = t_seg.get("end_time", 0)

        best_overlap = 0
        best_lang = t_seg.get("language", "en")

        for l_seg in lid_segments:
            l_start = l_seg.get("start_time", 0)
            l_end = l_seg.get("end_time", 0)

            overlap = max(0, min(t_end, l_end) - max(t_start, l_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_lang = l_seg.get("language", "unk")

        t_seg["language"] = best_lang

    return transcript_segments


def compute_wer_if_available(
    transcript_segments: List[Dict],
    ground_truth_path: Optional[str] = None,
) -> Optional[Dict[str, float]]:
    """Compute WER if ground truth transcript is available."""
    if ground_truth_path is None:
        return None

    gt_path = Path(ground_truth_path)
    if not gt_path.exists():
        logger.info("No ground truth found; skipping WER computation")
        return None

    try:
        from jiwer import wer, cer

        with open(gt_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)

        if isinstance(gt_data, dict):
            reference = gt_data.get("text", "")
        elif isinstance(gt_data, list):
            reference = " ".join(item.get("text", "") for item in gt_data)
        else:
            reference = str(gt_data)

        hypothesis = " ".join(seg.get("text", "") for seg in transcript_segments)

        if not reference or not hypothesis:
            return None

        wer_score = wer(reference, hypothesis)
        cer_score = cer(reference, hypothesis)

        metrics = {"wer": float(wer_score), "cer": float(cer_score)}
        logger.info(f"WER: {wer_score:.4f}, CER: {cer_score:.4f}")
        return metrics

    except ImportError:
        logger.warning("jiwer not installed; skipping WER computation")
        return None


def run_stt_pipeline(
    audio_path: str,
    config_path: str = "configs/dataset_config.yaml",
    output_dir: str = "outputs",
    denoise_method: str = "deepfilternet",
    lid_model_path: str = "models/lid/best_model.pt",
    whisper_model: str = "large-v3",
    ngram_lm_path: str = "data/processed/ngram/maithili_3gram.arpa",
    technical_vocab_path: str = "data/processed/ngram/technical_vocab.json",
    alpha: float = 0.3,
    beta: float = 3.0,
    ground_truth_path: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict:
    """Run the complete Part I STT pipeline.

    Pipeline stages:
      1. Denoise input audio
      2. Frame-level LID for language boundary detection
      3. Constrained Whisper decoding with n-gram LM biasing
      4. Merge LID + transcript and save final output

    Returns:
        Dict with pipeline results including transcript path and metrics.
    """
    audio_p = Path(audio_path)
    if not audio_p.exists():
        raise FileNotFoundError(f"Input audio not found: {audio_path}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    logger.info(f"Running STT pipeline on: {audio_path}")

    denoised_path = stage_denoise(audio_path, output_dir, method=denoise_method)

    lid_segments = stage_lid(denoised_path, model_path=lid_model_path, device=device)

    if not lid_segments:
        logger.info("No trained LID model — transcribing full audio without segmentation")

    lid_output_path = out_dir / "lid_segments.json"
    with open(lid_output_path, "w", encoding="utf-8") as f:
        json.dump({"segments": lid_segments, "num_segments": len(lid_segments)}, f, indent=2)
    logger.info(f"LID segments saved to {lid_output_path}")

    transcript_segments = stage_transcribe(
        audio_path=denoised_path,
        language_segments=lid_segments if lid_segments else None,
        ngram_lm_path=ngram_lm_path,
        technical_vocab_path=technical_vocab_path,
        model_name=whisper_model,
        alpha=alpha,
        beta=beta,
        device=device,
    )

    final_segments = merge_lid_and_transcript(lid_segments, transcript_segments)

    wer_metrics = compute_wer_if_available(final_segments, ground_truth_path)

    transcript_path = out_dir / "transcript_code_switched.json"
    output_data = {
        "audio_source": str(audio_path),
        "denoised_audio": str(denoised_path),
        "pipeline_config": {
            "denoise_method": denoise_method,
            "whisper_model": whisper_model,
            "ngram_alpha": alpha,
            "technical_beta": beta,
        },
        "lid_summary": {
            "num_segments": len(lid_segments),
            "languages": list(set(s.get("language", "unk") for s in lid_segments)),
        },
        "segments": final_segments,
        "num_segments": len(final_segments),
        "total_duration": max((s.get("end_time", 0) for s in final_segments), default=0.0),
    }

    if wer_metrics:
        output_data["evaluation"] = wer_metrics

    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Final transcript saved to {transcript_path}")

    return {
        "transcript_path": str(transcript_path),
        "lid_segments_path": str(lid_output_path),
        "denoised_audio_path": denoised_path,
        "num_segments": len(final_segments),
        "wer_metrics": wer_metrics,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Full Part I pipeline: Denoise -> LID -> Constrained Whisper Decoding"
    )
    parser.add_argument(
        "--audio", default="outputs/audio/original_segment.wav",
        help="Path to input lecture audio"
    )
    parser.add_argument(
        "--config", default="configs/dataset_config.yaml",
        help="Path to dataset/pipeline config YAML"
    )
    parser.add_argument("--output-dir", default="outputs/", help="Output directory")
    parser.add_argument(
        "--denoise-method", choices=["deepfilternet", "spectral"],
        default="deepfilternet", help="Denoising method"
    )
    parser.add_argument("--lid-model", default="models/lid/best_model.pt")
    parser.add_argument("--whisper-model", default="large-v3")
    parser.add_argument("--ngram-lm", default="data/processed/ngram/maithili_3gram.arpa")
    parser.add_argument("--technical-vocab", default="data/processed/ngram/technical_vocab.json")
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=3.0)
    parser.add_argument("--ground-truth", default=None, help="Optional ground truth transcript")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    results = run_stt_pipeline(
        audio_path=args.audio,
        config_path=args.config,
        output_dir=args.output_dir,
        denoise_method=args.denoise_method,
        lid_model_path=args.lid_model,
        whisper_model=args.whisper_model,
        ngram_lm_path=args.ngram_lm,
        technical_vocab_path=args.technical_vocab,
        alpha=args.alpha,
        beta=args.beta,
        ground_truth_path=args.ground_truth,
        device=args.device,
    )

    logger.info(f"Pipeline complete. Results: {json.dumps(results, indent=2)}")


if __name__ == "__main__":
    main()
