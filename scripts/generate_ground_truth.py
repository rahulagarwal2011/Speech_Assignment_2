"""
Generate a ground truth transcript with proper Hindi/English detection.

Uses Whisper to transcribe the audio in both Hindi and English for each
window, then picks the correct language based on Devanagari script presence.

Usage:
  python scripts/generate_ground_truth.py
  python scripts/generate_ground_truth.py --audio outputs/audio/original_segment.wav

Then manually review and correct the text for each segment.
After correction, re-run evaluation:
  python scripts/evaluate_all.py
"""

import argparse
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_ground_truth(
    audio_path: str = "outputs/audio/original_segment.wav",
    output_path: str = "data/ground_truth_transcript.json",
    whisper_model: str = "medium",
    window_sec: float = 10.0,
    hop_sec: float = 5.0,
    device: str = None,
):
    import whisper
    import numpy as np

    if device is None:
        device = "cpu"

    logger.info(f"Loading Whisper {whisper_model}...")
    model = whisper.load_model(whisper_model, device=device)

    logger.info(f"Loading audio: {audio_path}")
    audio = whisper.load_audio(audio_path)
    sr = 16000
    duration = len(audio) / sr
    logger.info(f"Audio duration: {duration:.1f}s")

    window_samples = int(window_sec * sr)
    hop_samples = int(hop_sec * sr)

    # Step 1: Detect language per window using script detection
    logger.info("Step 1: Detecting language per window...")
    window_langs = []
    total_windows = (len(audio) - int(sr * 2)) // hop_samples + 1

    for idx, start in enumerate(range(0, len(audio) - int(sr * 2), hop_samples)):
        end = min(start + window_samples, len(audio))
        chunk = audio[start:end]

        hi_result = whisper.transcribe(
            model, chunk,
            language="hi", task="transcribe",
            beam_size=1, best_of=1, without_timestamps=True,
        )
        hi_text = hi_result.get("text", "")

        devanagari = len(re.findall(r'[\u0900-\u097F]', hi_text))
        latin = len(re.findall(r'[a-zA-Z]', hi_text))
        total_chars = devanagari + latin

        if total_chars > 0 and devanagari / total_chars > 0.3:
            lang = "hi"
        else:
            lang = "en"

        window_langs.append({
            "start_time": round(start / sr, 2),
            "end_time": round(end / sr, 2),
            "language": lang,
        })

        if (idx + 1) % 10 == 0:
            logger.info(f"  {idx+1}/{total_windows} windows processed")

    # Step 2: Merge consecutive same-language windows
    merged = [window_langs[0].copy()]
    for w in window_langs[1:]:
        if w["language"] == merged[-1]["language"]:
            merged[-1]["end_time"] = w["end_time"]
        else:
            merged.append(w.copy())

    en_count = sum(1 for m in merged if m["language"] == "en")
    hi_count = sum(1 for m in merged if m["language"] == "hi")
    logger.info(f"Detected {len(merged)} segments: {en_count} English, {hi_count} Hindi")

    # Step 3: Transcribe each merged segment with correct language
    logger.info("Step 2: Transcribing each segment...")
    gt_segments = []
    seg_id = 0

    for i, lang_seg in enumerate(merged):
        start_sample = int(lang_seg["start_time"] * sr)
        end_sample = int(lang_seg["end_time"] * sr)
        seg_audio = audio[start_sample:end_sample]

        if len(seg_audio) < sr * 0.5:
            continue

        lang_code = lang_seg["language"]
        logger.info(
            f"  [{i+1}/{len(merged)}] {lang_seg['start_time']:.1f}s-"
            f"{lang_seg['end_time']:.1f}s (lang={lang_code})"
        )

        result = whisper.transcribe(
            model, seg_audio,
            language=lang_code, task="transcribe",
            beam_size=5, best_of=5, word_timestamps=True,
        )

        for seg in result.get("segments", []):
            text = seg["text"].strip()
            if not text:
                continue
            gt_segments.append({
                "id": seg_id,
                "start_time": round(lang_seg["start_time"] + seg["start"], 2),
                "end_time": round(lang_seg["start_time"] + seg["end"], 2),
                "language": lang_code,
                "text": text,
                "corrected": False,
            })
            seg_id += 1

    # Save
    gt = {
        "note": "Review and correct 'text' for each segment. Set 'corrected' to true after fixing. Hindi segments should have Devanagari text.",
        "audio_source": audio_path,
        "segments": gt_segments,
        "num_segments": len(gt_segments),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=2, ensure_ascii=False)

    en_segs = sum(1 for s in gt_segments if s["language"] == "en")
    hi_segs = sum(1 for s in gt_segments if s["language"] == "hi")
    logger.info(f"\nGround truth written to {output_path}")
    logger.info(f"  {len(gt_segments)} segments: {en_segs} English, {hi_segs} Hindi")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Open {output_path} and correct any errors")
    logger.info(f"  2. Run: python scripts/evaluate_all.py")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Generate ground truth transcript")
    parser.add_argument("--audio", default="outputs/audio/original_segment.wav")
    parser.add_argument("--output", default="data/ground_truth_transcript.json")
    parser.add_argument("--model", default="medium")
    parser.add_argument("--window", type=float, default=10.0)
    parser.add_argument("--hop", type=float, default=5.0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    generate_ground_truth(
        audio_path=args.audio,
        output_path=args.output,
        whisper_model=args.model,
        window_sec=args.window,
        hop_sec=args.hop,
        device=args.device,
    )
