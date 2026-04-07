"""
Part II orchestrator: Technical Dictionary -> IPA Conversion -> Maithili Translation.

Runs the three stages of the translation/IPA pipeline:
  1. Build/update the 500-word technical dictionary
  2. Convert code-switched transcript to IPA
  3. Translate transcript to Maithili
  4. Validate outputs

CLI: python scripts/run_translation_pipeline.py \
       --transcript outputs/transcript_code_switched.json \
       --config configs/dataset_config.yaml --output-dir outputs/
"""

import csv
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Optional

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    path = Path(config_path)
    if not path.exists():
        logger.warning(f"Config not found at {config_path}; using defaults")
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f)


def stage_build_dictionary(
    parallel_corpus_path: str,
    output_path: str,
    min_terms: int = 500,
) -> Dict:
    """Stage 1: Build or update the technical dictionary."""
    existing_count = 0
    out_p = Path(output_path)
    if out_p.exists():
        with open(out_p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            existing_count = sum(1 for _ in reader)

    if existing_count >= min_terms:
        logger.info(
            f"[Stage 1] Dictionary already has {existing_count} entries "
            f"(>= {min_terms}); skipping rebuild"
        )
        return {"status": "skipped", "existing_count": existing_count}

    logger.info(f"[Stage 1] Building dictionary (current: {existing_count}, target: {min_terms})")
    from scripts.build_technical_dictionary import build_dictionary

    dictionary = build_dictionary(parallel_corpus_path, output_path, min_terms)
    return {"status": "built", "count": len(dictionary)}


def stage_ipa_conversion(
    transcript_path: str,
    output_path: str,
    g2p_overrides_path: Optional[str] = None,
) -> Dict:
    """Stage 2: Convert transcript to IPA."""
    logger.info(f"[Stage 2] Converting transcript to IPA: {transcript_path}")
    from scripts.ipa_converter import HinglishIPAConverter

    converter = HinglishIPAConverter(g2p_overrides_path=g2p_overrides_path)
    result = converter.convert_transcript(transcript_path, output_path)
    return result


def stage_translate(
    transcript_path: str,
    output_path: str,
    dictionary_path: str,
    en_model: str = "ai4bharat/indictrans2-en-indic-1B",
    hi_model: str = "ai4bharat/indictrans2-indic-indic-1B",
) -> Dict:
    """Stage 3: Translate transcript to Maithili."""
    logger.info(f"[Stage 3] Translating to Maithili: {transcript_path}")
    from scripts.translate_to_maithili import MaithiliTranslator

    translator = MaithiliTranslator(
        en_model_name=en_model,
        hi_model_name=hi_model,
        dictionary_path=dictionary_path,
    )
    result = translator.translate_transcript(transcript_path, output_path)
    return result


def validate_outputs(
    dictionary_path: str,
    ipa_path: str,
    translation_path: str,
    min_terms: int = 500,
) -> Dict:
    """Stage 4: Validate all outputs meet requirements."""
    results = {"passed": True, "checks": []}

    dict_p = Path(dictionary_path)
    if dict_p.exists():
        with open(dict_p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            rows = list(reader)
        dict_count = len(rows)
        translated_count = sum(
            1 for r in rows
            if (r.get("maithili") or "")
            and not r.get("maithili", "").startswith("[TRANSLATE:")
        )
        check = {
            "name": "dictionary_size",
            "expected": f">= {min_terms}",
            "actual": dict_count,
            "passed": dict_count >= min_terms,
        }
        results["checks"].append(check)
        results["checks"].append({
            "name": "dictionary_translated",
            "actual": translated_count,
            "total": dict_count,
            "ratio": round(translated_count / max(dict_count, 1), 3),
        })
        if not check["passed"]:
            results["passed"] = False
    else:
        results["checks"].append({
            "name": "dictionary_exists", "passed": False, "path": dictionary_path,
        })
        results["passed"] = False

    ipa_p = Path(ipa_path)
    if ipa_p.exists():
        with open(ipa_p, "r", encoding="utf-8") as f:
            ipa_data = json.load(f)
        segments = ipa_data.get("segments", [])
        ipa_filled = sum(1 for s in segments if s.get("ipa", "").strip())
        total = len(segments)
        coverage = round(ipa_filled / max(total, 1), 3)
        results["checks"].append({
            "name": "ipa_coverage",
            "segments_with_ipa": ipa_filled,
            "total_segments": total,
            "coverage": coverage,
            "passed": coverage >= 0.8,
        })
        if coverage < 0.8:
            results["passed"] = False
    else:
        results["checks"].append({
            "name": "ipa_exists", "passed": False, "path": ipa_path,
        })
        results["passed"] = False

    trans_p = Path(translation_path)
    if trans_p.exists():
        with open(trans_p, "r", encoding="utf-8") as f:
            trans_data = json.load(f)
        segments = trans_data.get("segments", [])
        translated = sum(1 for s in segments if s.get("maithili_text", "").strip())
        total = len(segments)
        coverage = round(translated / max(total, 1), 3)
        results["checks"].append({
            "name": "translation_coverage",
            "segments_translated": translated,
            "total_segments": total,
            "coverage": coverage,
            "passed": coverage >= 0.8,
        })
        if coverage < 0.8:
            results["passed"] = False
    else:
        results["checks"].append({
            "name": "translation_exists", "passed": False, "path": translation_path,
        })
        results["passed"] = False

    return results


def run_translation_pipeline(
    transcript_path: str,
    config_path: str = "configs/dataset_config.yaml",
    output_dir: str = "outputs",
) -> Dict:
    """Run the complete Part II pipeline.

    Args:
        transcript_path: Path to transcript_code_switched.json from Part I.
        config_path: Path to dataset_config.yaml.
        output_dir: Directory for output files.

    Returns:
        Summary dict with results from each stage.
    """
    config = load_config(config_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tc = config.get("technical_dictionary", {})
    min_terms = tc.get("min_terms", 500)
    dict_output = tc.get("output_path", "data/manual/technical_parallel_corpus.tsv")
    parallel_corpus = "data/processed/translation/en_mai_parallel_clean.tsv"
    g2p_overrides = "data/processed/translation/maithili_grapheme_to_ipa.json"

    ipa_output = str(out_dir / "transcript_ipa.json")
    translation_output = str(out_dir / "maithili_translated_text.json")

    logger.info("=" * 60)
    logger.info("Part II Translation Pipeline")
    logger.info("=" * 60)

    dict_result = stage_build_dictionary(parallel_corpus, dict_output, min_terms)
    logger.info(f"Stage 1 result: {dict_result}")

    ipa_result = stage_ipa_conversion(transcript_path, ipa_output, g2p_overrides)
    logger.info(f"Stage 2 result: {ipa_result}")

    trans_result = stage_translate(
        transcript_path, translation_output, dict_output,
    )
    logger.info(f"Stage 3 result: {trans_result}")

    validation = validate_outputs(dict_output, ipa_output, translation_output, min_terms)
    logger.info(f"Stage 4 validation: {json.dumps(validation, indent=2)}")

    summary = {
        "dictionary": dict_result,
        "ipa_conversion": ipa_result,
        "translation": trans_result,
        "validation": validation,
    }

    summary_path = out_dir / "translation_pipeline_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info(f"Pipeline complete. Summary saved to {summary_path}")
    logger.info(f"Validation passed: {validation['passed']}")
    for check in validation["checks"]:
        status = "PASS" if check.get("passed", True) else "FAIL"
        logger.info(f"  [{status}] {check['name']}: {check}")
    logger.info("=" * 60)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Part II orchestrator: Dictionary -> IPA -> Maithili Translation"
    )
    parser.add_argument(
        "--transcript",
        default="outputs/transcript_code_switched.json",
        help="Path to code-switched transcript JSON from Part I",
    )
    parser.add_argument(
        "--config",
        default="configs/dataset_config.yaml",
        help="Path to dataset config YAML",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/",
        help="Output directory",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    results = run_translation_pipeline(args.transcript, args.config, args.output_dir)
    logger.info(f"Final results: {json.dumps(results, indent=2)}")


if __name__ == "__main__":
    main()
