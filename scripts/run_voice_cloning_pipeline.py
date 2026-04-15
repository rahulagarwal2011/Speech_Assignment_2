"""
Part III orchestrator for Speech Understanding PA-2: Voice Cloning Pipeline.

Steps:
  1. Extract speaker embedding from student's 60 s recording
  2. Extract professor prosody from denoised lecture
  3. Synthesise Maithili lecture with both Parler-TTS and MMS
  4. Compute MCD for each against the student reference; pick the lower
  5. Apply DTW prosody warping to the selected synthesis
  6. Also save a flat (unwarped) version for ablation
  7. Print comparison summary

All heavy I/O (model loading, synthesis) happens when the user actually
executes this script; the code itself performs no downloads.
"""

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional

import yaml

logger = logging.getLogger(__name__)

DEFAULTS = {
    "student_audio": "outputs/audio/student_voice_ref.wav",
    "professor_audio": "outputs/audio/original_segment_denoised.wav",
    "maithili_text": "outputs/maithili_translated_text.json",
    "embedding_out": "outputs/audio/your_voice_embedding.pt",
    "final_output": "outputs/audio/output_LRL_cloned.wav",
    "flat_output": "outputs/audio/output_LRL_flat.wav",
    "embedding_method": "speechbrain",
    "device": None,
}


def _check_file(path: str, label: str) -> bool:
    if not Path(path).exists():
        logger.error("%s not found: %s", label, path)
        return False
    return True


def run_voice_cloning_pipeline(
    config: Optional[Dict] = None,
    student_audio: Optional[str] = None,
    professor_audio: Optional[str] = None,
    maithili_text: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict:
    """Run the full Part III voice cloning pipeline.

    Returns a summary dict with MCD scores, selected model, and output paths.
    """
    # Resolve parameters from config / defaults
    cfg = config or {}
    student_audio = student_audio or cfg.get("student_audio", DEFAULTS["student_audio"])
    professor_audio = professor_audio or cfg.get("professor_audio", DEFAULTS["professor_audio"])
    maithili_text = maithili_text or cfg.get("maithili_text", DEFAULTS["maithili_text"])
    device = device or cfg.get("device", DEFAULTS["device"])
    if device is None:
        from scripts.device_utils import get_device
        device = get_device()
    embedding_out = cfg.get("embedding_out", DEFAULTS["embedding_out"])
    final_output = cfg.get("final_output", DEFAULTS["final_output"])
    flat_output = cfg.get("flat_output", DEFAULTS["flat_output"])
    emb_method = cfg.get("embedding_method", DEFAULTS["embedding_method"])

    # ------------------------------------------------------------------
    # 1. Validate prerequisites
    # ------------------------------------------------------------------
    ok = True
    for path, label in [
        (student_audio, "Student voice recording"),
        (professor_audio, "Professor denoised audio"),
        (maithili_text, "Maithili translated text"),
    ]:
        ok = _check_file(path, label) and ok
    if not ok:
        logger.error("Aborting – missing prerequisite files (see above).")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Extract speaker embedding
    # ------------------------------------------------------------------
    logger.info("Step 1/7: Extracting speaker embedding (%s) ...", emb_method)
    from scripts.extract_speaker_embedding import SpeakerEmbeddingExtractor

    extractor = SpeakerEmbeddingExtractor(method=emb_method, device=device)
    embedding = extractor.extract(student_audio)
    extractor.save_embedding(embedding, embedding_out)

    # ------------------------------------------------------------------
    # 3. Extract professor prosody (deferred to warper init below)
    # ------------------------------------------------------------------
    logger.info("Step 2/7: Professor prosody will be extracted during warping.")

    # ------------------------------------------------------------------
    # 4 & 5. Synthesise with both models (skip if cached)
    # ------------------------------------------------------------------
    from scripts.synthesize_maithili import MaithiliSynthesizer

    cached_synth = Path("outputs/audio/synth_xtts_raw.wav")
    if not cached_synth.exists():
        cached_synth = Path("outputs/audio/synth_mms_raw.wav")
    if cached_synth.exists():
        logger.info("Step 3/7: Found cached synthesis at %s — skipping re-synthesis", cached_synth)
        selected_path = str(cached_synth)
        best_model = "xtts" if "xtts" in cached_synth.name else "mms"
        mcd_scores = {}
        results = {best_model: {"path": str(cached_synth), "error": None}}
    else:
        tmp_dir = Path(tempfile.mkdtemp(prefix="voice_clone_"))
        xtts_path = str(tmp_dir / "synth_xtts.wav")
        mms_path = str(tmp_dir / "synth_mms.wav")

        results: Dict[str, Dict] = {}

        # Try XTTS first (actual voice cloning), fall back to MMS
        for model_name, out_path in [("xtts", xtts_path), ("mms", mms_path)]:
            logger.info("Step 3/7: Synthesising with %s ...", model_name)
            try:
                synth = MaithiliSynthesizer(
                    model_name=model_name,
                    speaker_embedding_path=embedding_out,
                    speaker_wav_path=student_audio,
                    device=device,
                )
                synth.synthesize_lecture(maithili_text, out_path)
                results[model_name] = {"path": out_path, "error": None}
            except Exception as e:
                logger.warning("Synthesis with %s failed: %s", model_name, e)
                results[model_name] = {"path": None, "error": str(e)}

        # ------------------------------------------------------------------
        # 6. Compute MCD and select the better model
        # ------------------------------------------------------------------
        logger.info("Step 4/7: Computing MCD for model selection ...")
        mcd_scores: Dict[str, float] = {}

        mcd_calculator = None
        for model_name, info in results.items():
            if info["path"] is not None and Path(info["path"]).exists():
                try:
                    if mcd_calculator is None:
                        mcd_calculator = MaithiliSynthesizer.__new__(MaithiliSynthesizer)
                        mcd_calculator.model_name = model_name
                    mcd = mcd_calculator.compute_mcd(student_audio, info["path"])
                    mcd_scores[model_name] = mcd
                except Exception as e:
                    logger.warning("MCD computation failed for %s: %s", model_name, e)

        if mcd_scores:
            best_model = min(mcd_scores, key=mcd_scores.get)
        else:
            best_model = next(
                (m for m, r in results.items() if r["path"] is not None), "parler",
            )
        logger.info("MCD scores: %s  ->  selected '%s'", mcd_scores, best_model)

        selected_path = results.get(best_model, {}).get("path")
        if selected_path is None or not Path(selected_path).exists():
            fallback = next(
                (r["path"] for r in results.values() if r["path"] and Path(r["path"]).exists()),
                None,
            )
            if fallback is None:
                logger.error("Both TTS models failed – cannot continue.")
                sys.exit(1)
            selected_path = fallback
            best_model = [m for m, r in results.items() if r["path"] == fallback][0]

    # ------------------------------------------------------------------
    # 7. Prosody warping
    # ------------------------------------------------------------------
    logger.info("Step 5/7: Applying DTW prosody warping ...")
    from scripts.prosody_warping import ProsodyWarper

    warper = ProsodyWarper(professor_audio)
    warper.warp(selected_path, final_output)

    logger.info("Step 6/7: Saving flat (unwarped) ablation version ...")
    warper.warp_flat(selected_path, flat_output)

    # ------------------------------------------------------------------
    # 8. Summary
    # ------------------------------------------------------------------
    summary = {
        "selected_model": best_model,
        "mcd_scores": mcd_scores,
        "output_warped": final_output,
        "output_flat": flat_output,
        "embedding_path": embedding_out,
        "embedding_method": emb_method,
        "synth_results": {m: {"error": r["error"]} for m, r in results.items()},
    }

    summary_path = Path("outputs/voice_cloning_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Step 7/7: Pipeline complete.")
    logger.info("=" * 60)
    logger.info("  Selected TTS model : %s", best_model)
    for m, s in mcd_scores.items():
        logger.info("  MCD (%s)           : %.3f dB", m, s)
    logger.info("  Warped output      : %s", final_output)
    logger.info("  Flat output        : %s", flat_output)
    logger.info("  Summary            : %s", summary_path)
    logger.info("=" * 60)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Part III orchestrator: voice cloning pipeline",
    )
    parser.add_argument(
        "--config", type=str, default="configs/dataset_config.yaml",
        help="Path to dataset config YAML",
    )
    parser.add_argument(
        "--student-audio", type=str,
        default=DEFAULTS["student_audio"],
        help="Path to student's 60 s voice recording",
    )
    parser.add_argument(
        "--professor-audio", type=str,
        default=DEFAULTS["professor_audio"],
        help="Path to professor's denoised lecture audio",
    )
    parser.add_argument(
        "--maithili-text", type=str,
        default=DEFAULTS["maithili_text"],
        help="Path to Maithili translated text JSON",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Torch device (auto-detects cuda/mps/cpu)",
    )
    args = parser.parse_args()

    cfg = {}
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}

    run_voice_cloning_pipeline(
        config=cfg,
        student_audio=args.student_audio,
        professor_audio=args.professor_audio,
        maithili_text=args.maithili_text,
        device=args.device,
    )


if __name__ == "__main__":
    main()
