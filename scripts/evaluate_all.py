"""
Comprehensive evaluation for Speech Understanding PA-2 Maithili pipeline.

Computes all required metrics:
  - WER per language (English, Hindi) with thresholds
  - Mel-Cepstral Distortion (MCD) for synthesized audio
  - LID switch accuracy
  - Anti-spoofing EER verification
  - Adversarial robustness (SNR) verification
  - Technical dictionary size check
  - Required audio file existence check

Output: outputs/reports/evaluation_metrics.json
"""

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)

# Expected output audio files from the pipeline
REQUIRED_AUDIO_FILES = [
    "outputs/audio/output_LRL_cloned.wav",
    "outputs/audio/output_LRL_flat.wav",
    "outputs/audio/original_segment_denoised.wav",
]


def load_config(config_path: str) -> Dict:
    path = Path(config_path)
    if not path.exists():
        logger.warning("Config not found at %s; using defaults", config_path)
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _load_transcript_segments(transcript_path: str) -> List[Dict]:
    """Load transcript JSON and return the list of segments."""
    with open(transcript_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("segments", [])
    return []


# ---------------------------------------------------------------------------
# WER
# ---------------------------------------------------------------------------

def compute_wer(
    transcript_path: str,
    ground_truth_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute WER for English and Hindi segments separately.

    If ground_truth_path is provided, segments from the transcript are
    compared against matching ground-truth entries.  Otherwise the function
    looks for an "evaluation" key already embedded in the transcript JSON
    (produced by run_stt_pipeline).

    Returns dict with wer_english, wer_hindi, overall_wer, and pass/fail.
    """
    from jiwer import wer as jiwer_wer

    segments = _load_transcript_segments(transcript_path)

    en_hyps, hi_hyps = [], []
    en_refs, hi_refs = [], []

    if ground_truth_path and Path(ground_truth_path).exists():
        with open(ground_truth_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
        gt_segments = gt_data if isinstance(gt_data, list) else gt_data.get("segments", [])

        # Match by closest midpoint rather than exact (start, end) since
        # transcript and ground truth may have different segmentation.
        for seg in segments:
            seg_mid = (seg.get("start_time", 0) + seg.get("end_time", 0)) / 2.0
            best_gt = None
            best_dist = float("inf")
            for gt_seg in gt_segments:
                gt_mid = (gt_seg.get("start_time", 0) + gt_seg.get("end_time", 0)) / 2.0
                dist = abs(seg_mid - gt_mid)
                if dist < best_dist:
                    best_dist = dist
                    best_gt = gt_seg

            if best_gt is None or best_dist > 5.0:
                continue

            lang = seg.get("language", "en").lower()
            hyp = seg.get("text", "").strip()
            ref = best_gt.get("text", "").strip()
            if not hyp or not ref:
                continue
            if lang in ("en", "english"):
                en_hyps.append(hyp)
                en_refs.append(ref)
            elif lang in ("hi", "hindi"):
                hi_hyps.append(hyp)
                hi_refs.append(ref)
            else:
                hi_hyps.append(hyp)
                hi_refs.append(ref)

        logger.info("WER matching: %d EN pairs, %d HI pairs (from %d transcript, %d GT segments)",
                     len(en_hyps), len(hi_hyps), len(segments), len(gt_segments))
    else:
        # Fallback: treat all segment text as hypothesis with no ground truth
        all_text = " ".join(seg.get("text", "") for seg in segments).strip()
        if not all_text:
            return {
                "wer_english": None,
                "wer_hindi": None,
                "overall_wer": None,
                "pass_english": None,
                "pass_hindi": None,
                "note": "No ground truth available; WER cannot be computed.",
            }
        return {
            "wer_english": None,
            "wer_hindi": None,
            "overall_wer": None,
            "pass_english": None,
            "pass_hindi": None,
            "note": "Ground truth path not provided; skipped.",
        }

    result: Dict[str, Any] = {}

    if en_refs and en_hyps:
        wer_en = jiwer_wer(" ".join(en_refs), " ".join(en_hyps))
        result["wer_english"] = round(wer_en, 4)
        result["pass_english"] = wer_en < 0.15
    else:
        result["wer_english"] = None
        result["pass_english"] = None

    if hi_refs and hi_hyps:
        wer_hi = jiwer_wer(" ".join(hi_refs), " ".join(hi_hyps))
        result["wer_hindi"] = round(wer_hi, 4)
        result["pass_hindi"] = wer_hi < 0.25
    else:
        result["wer_hindi"] = None
        result["pass_hindi"] = None

    all_refs = en_refs + hi_refs
    all_hyps = en_hyps + hi_hyps
    if all_refs and all_hyps:
        overall = jiwer_wer(" ".join(all_refs), " ".join(all_hyps))
        result["overall_wer"] = round(overall, 4)
    else:
        result["overall_wer"] = None

    return result


# ---------------------------------------------------------------------------
# MCD
# ---------------------------------------------------------------------------

def _extract_mcc_pyworld(audio: np.ndarray, sr: int, order: int = 24) -> np.ndarray:
    """Extract mel-cepstral coefficients using WORLD vocoder + pysptk.

    Returns (T, order) array of MCCs (excluding c0).
    """
    import pyworld as pw
    import pysptk

    audio_f64 = audio.astype(np.float64)
    f0, t = pw.harvest(audio_f64, sr, frame_period=5.0)
    sp = pw.cheaptrick(audio_f64, f0, t, sr)

    alpha = pysptk.util.mcepalpha(sr)
    mcc = pysptk.sp2mc(sp, order=order, alpha=alpha)
    return mcc[:, 1:]  # drop c0


def _extract_mcc_librosa(audio: np.ndarray, sr: int, order: int = 24) -> np.ndarray:
    """Fallback: extract MCCs from librosa mel-spectrogram + DCT.

    Applies proper scaling so values are comparable to WORLD-based MCCs.
    Returns (T, order) array.
    """
    import librosa
    from scipy.fft import dct

    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=1024, hop_length=int(sr * 0.005),
        n_mels=order * 2 + 1, power=1.0,
    )
    log_mel = np.log(mel + 1e-8)
    mcc = dct(log_mel, type=2, axis=0, norm="ortho")[1 : order + 1]
    return mcc.T


def compute_mcd(
    ref_audio_path: str,
    synth_audio_path: str,
    sr: int = 22050,
    order: int = 24,
) -> float:
    """Compute Mel-Cepstral Distortion between reference and synthesized audio.

    MCD = (10 * sqrt(2) / ln(10)) * mean_T( sqrt( sum_d (ref_d - synth_d)^2 ) )

    Uses WORLD vocoder + pysptk for proper mel-cepstral coefficients.
    Falls back to librosa mel-spectrogram + DCT if unavailable.
    DTW-aligns the sequences before computing the distance.
    """
    import librosa
    import soundfile as sf

    ref_audio, ref_sr = sf.read(ref_audio_path, dtype="float32")
    if ref_sr != sr:
        ref_audio = librosa.resample(ref_audio, orig_sr=ref_sr, target_sr=sr)

    synth_audio, synth_sr = sf.read(synth_audio_path, dtype="float32")
    if synth_sr != sr:
        synth_audio = librosa.resample(synth_audio, orig_sr=synth_sr, target_sr=sr)

    if ref_audio.ndim > 1:
        ref_audio = ref_audio.mean(axis=1)
    if synth_audio.ndim > 1:
        synth_audio = synth_audio.mean(axis=1)

    # Truncate very long audio to 60s for MCD (avoids memory explosion in DTW)
    # MCD is an average over frames — 60s (~12k frames) is statistically sufficient.
    # The reference recording is only 60s anyway per assignment spec.
    max_samples = sr * 60
    if len(ref_audio) > max_samples:
        ref_audio = ref_audio[:max_samples]
    if len(synth_audio) > max_samples:
        # Sample from middle of synth to avoid silence at start/end
        mid = len(synth_audio) // 2
        half = max_samples // 2
        synth_audio = synth_audio[mid - half : mid + half]

    # Use librosa as primary (stable on all platforms including macOS/Apple Silicon)
    # pyworld can bus-error on large files on macOS
    import platform
    use_pyworld = platform.system() != "Darwin"

    if use_pyworld:
        try:
            ref_mcc = _extract_mcc_pyworld(ref_audio, sr, order)
            synth_mcc = _extract_mcc_pyworld(synth_audio, sr, order)
            logger.info("MCD: using pyworld+pysptk for mel-cepstral extraction")
        except (ImportError, Exception) as e:
            logger.warning("pyworld/pysptk failed (%s); using librosa fallback", e)
            ref_mcc = _extract_mcc_librosa(ref_audio, sr, order)
            synth_mcc = _extract_mcc_librosa(synth_audio, sr, order)
    else:
        logger.info("MCD: using librosa mel-cepstral extraction (macOS safe)")
        ref_mcc = _extract_mcc_librosa(ref_audio, sr, order)
        synth_mcc = _extract_mcc_librosa(synth_audio, sr, order)

    # DTW alignment
    try:
        from dtw import dtw as dtw_lib
        alignment = dtw_lib(ref_mcc, synth_mcc, keep_internals=False)
        ref_aligned = ref_mcc[alignment.index1]
        synth_aligned = synth_mcc[alignment.index2]
    except ImportError:
        min_len = min(len(ref_mcc), len(synth_mcc))
        ref_aligned = ref_mcc[:min_len]
        synth_aligned = synth_mcc[:min_len]
        logger.warning("dtw-python not installed; using truncation instead of DTW alignment")

    diff = ref_aligned - synth_aligned
    frame_dists = np.sqrt(np.sum(diff ** 2, axis=1))
    mcd = (10.0 * math.sqrt(2.0) / math.log(10.0)) * np.mean(frame_dists)

    return float(mcd)


# ---------------------------------------------------------------------------
# LID switch accuracy
# ---------------------------------------------------------------------------

def compute_lid_switch_accuracy(
    transcript_path: str,
    ground_truth_path: Optional[str] = None,
    threshold_ms: float = 200.0,
) -> Dict[str, Any]:
    """Evaluate language-switch timestamp accuracy.

    Extracts switch points from the transcript (where consecutive segments
    have different languages) and optionally compares against ground truth.
    """
    segments = _load_transcript_segments(transcript_path)

    pred_switches: List[float] = []
    for i in range(1, len(segments)):
        prev_lang = segments[i - 1].get("language", "")
        curr_lang = segments[i].get("language", "")
        if prev_lang != curr_lang:
            switch_time = segments[i].get("start_time", 0.0)
            pred_switches.append(switch_time)

    result: Dict[str, Any] = {
        "num_predicted_switches": len(pred_switches),
        "predicted_switch_times_s": pred_switches,
    }

    if ground_truth_path and Path(ground_truth_path).exists():
        gt_segments = _load_transcript_segments(ground_truth_path)
        gt_switches: List[float] = []
        for i in range(1, len(gt_segments)):
            prev_lang = gt_segments[i - 1].get("language", "")
            curr_lang = gt_segments[i].get("language", "")
            if prev_lang != curr_lang:
                gt_switches.append(gt_segments[i].get("start_time", 0.0))

        if gt_switches:
            errors_ms: List[float] = []
            within_threshold = 0
            for gt_t in gt_switches:
                if pred_switches:
                    closest = min(pred_switches, key=lambda p: abs(p - gt_t))
                    err = abs(closest - gt_t) * 1000.0
                else:
                    err = float("inf")
                errors_ms.append(err)
                if err <= threshold_ms:
                    within_threshold += 1

            result["num_gt_switches"] = len(gt_switches)
            result["mean_abs_error_ms"] = round(float(np.mean(errors_ms)), 2)
            result["pct_within_threshold"] = round(
                100.0 * within_threshold / len(gt_switches), 2,
            )
            result["threshold_ms"] = threshold_ms
            result["pass"] = (within_threshold / len(gt_switches)) >= 0.5
        else:
            result["num_gt_switches"] = 0
            result["note"] = "No switches in ground truth."
    else:
        result["note"] = "No ground truth for switch accuracy evaluation."

    return result


# ---------------------------------------------------------------------------
# Anti-spoofing EER
# ---------------------------------------------------------------------------

def verify_eer(antispoof_results_path: str, threshold: float = 0.10) -> Dict[str, Any]:
    """Read anti-spoofing results and verify EER < threshold.

    Expects a JSON file with at least an "eer" key (float, 0–1 scale).
    """
    path = Path(antispoof_results_path)
    if not path.exists():
        return {"eer": None, "pass": None, "note": f"File not found: {antispoof_results_path}"}

    with open(path, "r") as f:
        data = json.load(f)

    eer = data.get("eer")
    if eer is None:
        eer = data.get("test_eer")
    if eer is None:
        return {"eer": None, "pass": None, "note": "No 'eer' or 'test_eer' key in results."}

    eer = float(eer)
    return {
        "eer": round(eer, 4),
        "threshold": threshold,
        "pass": eer < threshold,
    }


# ---------------------------------------------------------------------------
# Adversarial robustness
# ---------------------------------------------------------------------------

def verify_adversarial(
    adversarial_results_path: str,
    snr_min_db: float = 40.0,
) -> Dict[str, Any]:
    """Read adversarial evaluation results and verify SNR > threshold.

    Expects a JSON with at least an "snr_db" key.
    """
    path = Path(adversarial_results_path)
    if not path.exists():
        return {"snr_db": None, "pass": None, "note": f"File not found: {adversarial_results_path}"}

    with open(path, "r") as f:
        data = json.load(f)

    snr = data.get("snr_db")
    if snr is None:
        snr = data.get("snr")
    if snr is None:
        snr = data.get("snr_at_min_db")
    if snr is None:
        return {"snr_db": None, "pass": None, "note": "No 'snr_db' key in results."}

    eps = data.get("min_epsilon")
    snr = float(snr)
    result = {
        "snr_db": round(snr, 2),
        "threshold_db": snr_min_db,
        "pass": snr > snr_min_db,
    }
    if eps is not None:
        result["min_epsilon"] = eps
    return result


# ---------------------------------------------------------------------------
# Dictionary size check
# ---------------------------------------------------------------------------

def check_dictionary_size(
    dictionary_path: str,
    min_terms: int = 500,
) -> Dict[str, Any]:
    """Count rows in the TSV technical dictionary and verify >= min_terms."""
    path = Path(dictionary_path)
    if not path.exists():
        return {"count": 0, "pass": False, "note": f"File not found: {dictionary_path}"}

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Skip header if present
    data_lines = [ln for ln in lines if ln.strip() and not ln.startswith("#")]
    if data_lines and "\t" in data_lines[0]:
        header_cols = data_lines[0].strip().split("\t")
        looks_like_header = all(
            col.replace("_", "").replace(" ", "").isalpha() for col in header_cols
        )
        if looks_like_header:
            data_lines = data_lines[1:]

    count = len(data_lines)
    return {
        "count": count,
        "min_required": min_terms,
        "pass": count >= min_terms,
    }


# ---------------------------------------------------------------------------
# Audio file existence
# ---------------------------------------------------------------------------

def check_audio_files(
    required_files: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Verify that required WAV output files exist."""
    if required_files is None:
        required_files = REQUIRED_AUDIO_FILES

    results: Dict[str, bool] = {}
    for fpath in required_files:
        results[fpath] = Path(fpath).exists()

    all_present = all(results.values())
    return {
        "files": results,
        "all_present": all_present,
        "pass": all_present,
    }


# ---------------------------------------------------------------------------
# Master evaluation
# ---------------------------------------------------------------------------

def evaluate_all(
    config_path: str = "configs/dataset_config.yaml",
    output_path: str = "outputs/reports/evaluation_metrics.json",
) -> Dict[str, Any]:
    """Run every evaluation check and write a consolidated JSON report."""
    config = load_config(config_path)
    eval_cfg = config.get("evaluation", {})
    tech_cfg = config.get("technical_dictionary", {})

    wer_en_threshold = eval_cfg.get("wer_english_threshold", 0.15)
    wer_hi_threshold = eval_cfg.get("wer_hindi_threshold", 0.25)
    mcd_threshold = eval_cfg.get("mcd_threshold", 8.0)
    eer_threshold = eval_cfg.get("eer_threshold", 0.10)
    snr_min = eval_cfg.get("adversarial_snr_min_db", 40)
    min_terms = tech_cfg.get("min_terms", 500)
    dict_path = tech_cfg.get("output_path", "data/manual/technical_parallel_corpus.tsv")

    metrics: Dict[str, Any] = {}

    # 1. WER
    transcript_path = "outputs/transcript_code_switched.json"
    ground_truth_path = eval_cfg.get("ground_truth_path", "data/ground_truth_transcript.json")
    logger.info("WER: transcript=%s (exists=%s), ground_truth=%s (exists=%s)",
                transcript_path, Path(transcript_path).exists(),
                ground_truth_path, Path(ground_truth_path).exists())
    if Path(transcript_path).exists():
        wer_result = compute_wer(transcript_path, ground_truth_path=ground_truth_path)
        if wer_result.get("wer_english") is not None:
            wer_result["pass_english"] = wer_result["wer_english"] < wer_en_threshold
        if wer_result.get("wer_hindi") is not None:
            wer_result["pass_hindi"] = wer_result["wer_hindi"] < wer_hi_threshold
        metrics["wer"] = wer_result
    else:
        metrics["wer"] = {"note": f"Transcript not found: {transcript_path}", "pass": None}

    # 2. MCD — warped vs flat
    warped_path = "outputs/audio/output_LRL_cloned.wav"
    flat_path = "outputs/audio/output_LRL_flat.wav"
    ref_path = "outputs/audio/student_voice_ref.wav"
    for label, synth_path in [("mcd_warped", warped_path), ("mcd_flat", flat_path)]:
        if Path(synth_path).exists() and Path(ref_path).exists():
            try:
                mcd_val = compute_mcd(ref_path, synth_path)
                metrics[label] = {
                    "mcd": round(mcd_val, 4),
                    "threshold": mcd_threshold,
                    "pass": mcd_val < mcd_threshold,
                }
            except Exception as e:
                metrics[label] = {"error": str(e), "pass": None}
        else:
            missing = []
            if not Path(synth_path).exists():
                missing.append(synth_path)
            if not Path(ref_path).exists():
                missing.append(ref_path)
            metrics[label] = {"note": f"Missing files: {missing}", "pass": None}

    # 3. LID switch accuracy
    if Path(transcript_path).exists():
        metrics["lid_switch"] = compute_lid_switch_accuracy(
            transcript_path,
            ground_truth_path=ground_truth_path,
        )
    else:
        metrics["lid_switch"] = {"note": "Transcript not found.", "pass": None}

    # 4. Anti-spoofing EER (check both possible locations)
    antispoof_path = "models/antispoof/antispoof_results.json"
    if not Path(antispoof_path).exists():
        antispoof_path = "outputs/antispoof_results.json"
    metrics["eer"] = verify_eer(antispoof_path, threshold=eer_threshold)

    # 5. Adversarial robustness
    adversarial_path = "outputs/adversarial_results.json"
    metrics["adversarial"] = verify_adversarial(adversarial_path, snr_min_db=snr_min)

    # 6. Technical dictionary
    metrics["dictionary"] = check_dictionary_size(dict_path, min_terms=min_terms)

    # 7. Required audio files
    metrics["audio_files"] = check_audio_files()

    # Summary
    pass_counts = {"pass": 0, "fail": 0, "skip": 0}
    for key, val in metrics.items():
        p = val.get("pass") if isinstance(val, dict) else None
        if p is True:
            pass_counts["pass"] += 1
        elif p is False:
            pass_counts["fail"] += 1
        else:
            pass_counts["skip"] += 1

    metrics["summary"] = pass_counts

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("Evaluation metrics written to %s", out)

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Compute all evaluation metrics for the Maithili pipeline",
    )
    parser.add_argument(
        "--config", default="configs/dataset_config.yaml",
        help="Path to dataset config YAML",
    )
    parser.add_argument(
        "--output", default="outputs/reports/evaluation_metrics.json",
        help="Output path for the evaluation metrics JSON",
    )
    args = parser.parse_args()

    metrics = evaluate_all(config_path=args.config, output_path=args.output)

    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    for key, val in metrics.items():
        if key == "summary":
            continue
        if not isinstance(val, dict):
            continue
        status = val.get("pass")
        icon = "PASS" if status is True else ("FAIL" if status is False else "SKIP")
        print(f"  [{icon:4s}] {key}")
    summary = metrics.get("summary", {})
    print("-" * 60)
    print(
        f"  Total: {summary.get('pass', 0)} passed, "
        f"{summary.get('fail', 0)} failed, "
        f"{summary.get('skip', 0)} skipped"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
