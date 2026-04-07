"""
Phase 7 orchestrator: run all evaluation, figure generation, and report scaffolding.

Steps:
  1. Verify prerequisite manifests from phases 0–6
  2. Compute all metrics (evaluate_all)
  3. Generate all 7 report figures
  4. Scaffold the report and implementation note
  5. Verify required audio files exist
  6. Print pass/fail summary table
  7. Write phase7-evaluator-reporter.json manifest
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import yaml

logger = logging.getLogger(__name__)

PREREQUISITE_MANIFESTS = [
    "outputs/manifests/phase0-scaffolder.json",
    "outputs/manifests/phase1-data-downloader.json",
    "outputs/manifests/phase2-data-preparer.json",
    "outputs/manifests/phase3-stt-pipeline.json",
    "outputs/manifests/phase4-translation-ipa.json",
    "outputs/manifests/phase5-voice-cloning.json",
]

PHASE7_MANIFEST_PATH = "outputs/manifests/phase7-evaluator-reporter.json"


def verify_prerequisites() -> Dict[str, bool]:
    """Check that all prerequisite phase manifests exist and are complete."""
    results = {}
    for manifest_path in PREREQUISITE_MANIFESTS:
        p = Path(manifest_path)
        if not p.exists():
            results[manifest_path] = False
            continue
        try:
            with open(p, "r") as f:
                data = json.load(f)
            results[manifest_path] = data.get("status") == "complete"
        except (json.JSONDecodeError, KeyError):
            results[manifest_path] = False
    return results


def run_evaluation_pipeline(
    config_path: str = "configs/dataset_config.yaml",
) -> Dict:
    """Execute the full evaluation pipeline."""

    # ------------------------------------------------------------------
    # 1. Verify prerequisites
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("  Phase 7: Evaluation Pipeline")
    logger.info("=" * 60)

    logger.info("\n[Step 1/7] Verifying prerequisite manifests...")
    prereqs = verify_prerequisites()
    all_ok = all(prereqs.values())

    for path, ok in prereqs.items():
        status = "OK" if ok else "MISSING/INCOMPLETE"
        logger.info("  %s: %s", Path(path).stem, status)

    if not all_ok:
        missing = [p for p, ok in prereqs.items() if not ok]
        logger.warning(
            "Some prerequisite phases are missing: %s. "
            "Evaluation will proceed but some metrics may be unavailable.",
            ", ".join(Path(p).stem for p in missing),
        )

    # ------------------------------------------------------------------
    # 2. Compute all metrics
    # ------------------------------------------------------------------
    logger.info("\n[Step 2/7] Computing evaluation metrics...")
    from scripts.evaluate_all import evaluate_all

    metrics = evaluate_all(
        config_path=config_path,
        output_path="outputs/reports/evaluation_metrics.json",
    )

    # ------------------------------------------------------------------
    # 3. Generate figures
    # ------------------------------------------------------------------
    logger.info("\n[Step 3/7] Generating report figures...")
    from scripts.generate_figures import generate_all_figures

    try:
        figures = generate_all_figures(
            data_dir="outputs",
            output_dir="outputs/reports/figures",
        )
        logger.info("  Generated %d figures", len(figures))
    except Exception as e:
        logger.warning("Figure generation failed: %s", e)
        figures = {}

    # ------------------------------------------------------------------
    # 4. Scaffold report
    # ------------------------------------------------------------------
    logger.info("\n[Step 4/7] Scaffolding report templates...")
    from scripts.scaffold_report import (
        generate_implementation_note,
        generate_report_scaffold,
    )

    report_path = generate_report_scaffold("outputs/reports/PA2_Report_Scaffold.md")
    note_path = generate_implementation_note("outputs/reports/implementation_note.md")

    # ------------------------------------------------------------------
    # 5. Check required audio files
    # ------------------------------------------------------------------
    logger.info("\n[Step 5/7] Checking required audio files...")
    audio_status = metrics.get("audio_files", {})
    files_info = audio_status.get("files", {})
    for fpath, exists in files_info.items():
        logger.info("  %s: %s", fpath, "EXISTS" if exists else "MISSING")

    # ------------------------------------------------------------------
    # 6. Print summary table
    # ------------------------------------------------------------------
    logger.info("\n[Step 6/7] Pass / Fail Summary")
    logger.info("-" * 60)

    summary_rows: List[Dict] = []

    metric_labels = {
        "wer": "WER",
        "mcd_warped": "MCD (warped)",
        "mcd_flat": "MCD (flat/ablation)",
        "lid_switch": "LID switch accuracy",
        "eer": "Anti-spoofing EER",
        "adversarial": "Adversarial SNR",
        "dictionary": "Technical dictionary",
        "audio_files": "Required audio files",
    }

    for key, label in metric_labels.items():
        val = metrics.get(key, {})
        if not isinstance(val, dict):
            continue

        status = val.get("pass")
        if status is True:
            tag = "PASS"
        elif status is False:
            tag = "FAIL"
        else:
            tag = "SKIP"

        detail = ""
        if key == "wer":
            en = val.get("wer_english")
            hi = val.get("wer_hindi")
            if en is not None:
                detail += f"EN={en*100:.1f}% "
            if hi is not None:
                detail += f"HI={hi*100:.1f}%"
            # WER has separate pass for each language
            pass_en = val.get("pass_english")
            pass_hi = val.get("pass_hindi")
            if pass_en is False or pass_hi is False:
                tag = "FAIL"
            elif pass_en is True and pass_hi is True:
                tag = "PASS"
            else:
                tag = "SKIP"
        elif key in ("mcd_warped", "mcd_flat"):
            mcd = val.get("mcd")
            if mcd is not None:
                detail = f"{mcd:.2f} dB"
        elif key == "eer":
            eer = val.get("eer")
            if eer is not None:
                detail = f"{eer*100:.2f}%"
        elif key == "adversarial":
            snr = val.get("snr_db")
            if snr is not None:
                detail = f"{snr:.1f} dB"
        elif key == "dictionary":
            count = val.get("count")
            if count is not None:
                detail = f"{count} terms"

        logger.info("  [%4s] %-25s %s", tag, label, detail)
        summary_rows.append({"metric": label, "status": tag, "detail": detail})

    summary = metrics.get("summary", {})
    logger.info("-" * 60)
    logger.info(
        "  Total: %d passed, %d failed, %d skipped",
        summary.get("pass", 0),
        summary.get("fail", 0),
        summary.get("skip", 0),
    )
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 7. Write phase7 manifest
    # ------------------------------------------------------------------
    logger.info("\n[Step 7/7] Writing phase7 manifest...")
    manifest = {
        "agent": "phase7-evaluator-reporter",
        "status": "complete",
        "outputs": {
            "evaluation_metrics": "outputs/reports/evaluation_metrics.json",
            "figures_dir": "outputs/reports/figures/",
            "report_scaffold": report_path,
            "implementation_note": note_path,
            "figures_generated": list(figures.keys()) if figures else [],
        },
        "config_used": {"config_path": config_path},
        "metrics": {
            "summary": summary,
            "prerequisite_check": {p: ok for p, ok in prereqs.items()},
        },
    }

    manifest_out = Path(PHASE7_MANIFEST_PATH)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_out, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest written to %s", manifest_out)

    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Phase 7: Run full evaluation pipeline",
    )
    parser.add_argument(
        "--config", default="configs/dataset_config.yaml",
        help="Path to dataset config YAML",
    )
    args = parser.parse_args()

    run_evaluation_pipeline(config_path=args.config)


if __name__ == "__main__":
    main()
