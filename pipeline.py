"""
Speech Understanding PA-2: Maithili Pipeline — Single Entry Point

Run the entire pipeline end-to-end with a single command:

    python pipeline.py --all
    python pipeline.py --resume          # restart from last checkpoint
    python pipeline.py --step stt        # run a single step
    python pipeline.py --list            # show all steps

Constraints: Python + PyTorch only. No non-Python tooling at runtime.
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Optional

# Load .env before anything else reads os.environ
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; rely on shell-exported vars

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

CONFIG_PATH = "configs/dataset_config.yaml"
CHECKPOINT_PATH = Path("outputs/manifests/pipeline_checkpoint.json")


def _is_mps_available() -> bool:
    """Check if MPS backend is available without importing torch at module level."""
    try:
        import torch
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except ImportError:
        return False


# ── progress bar (pure Python, no external deps) ────────────────────────

def progress_bar(current: int, total: int, label: str = "", width: int = 40):
    """Print an inline progress bar to stderr."""
    filled = int(width * current / max(total, 1))
    bar = "█" * filled + "░" * (width - filled)
    pct = 100 * current / max(total, 1)
    print(f"\r  {bar} {pct:5.1f}%  {label}", end="", file=sys.stderr, flush=True)
    if current >= total:
        print(file=sys.stderr)


# ── checkpoint helpers ──────────────────────────────────────────────────

def load_checkpoint() -> Dict:
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, "r") as f:
            return json.load(f)
    return {"completed_steps": [], "results": {}}


def save_checkpoint(state: Dict):
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ── individual step wrappers (import at call time to keep startup fast) ─

def step_download(config_path: str) -> Dict:
    """Download all 9 datasets from HuggingFace (requires HF_TOKEN)."""
    from scripts.download_datasets import download_all

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.warning("HF_TOKEN not set — downloads requiring auth will fail")
    results = download_all(hf_token)
    ok = sum(1 for v in results.values() if v)
    return {"datasets_ok": ok, "datasets_total": len(results), "details": results}


def step_prepare_lid(config_path: str) -> Dict:
    """Prepare LID training data (Task 1.1)."""
    from scripts.prepare_lid_data import prepare_lid_data

    prepare_lid_data(config_path)
    return {"output_dir": "data/processed/lid"}


def step_prepare_ngram(config_path: str) -> Dict:
    """Prepare N-gram LM and technical vocabulary (Task 1.2)."""
    from scripts.prepare_ngram_data import prepare_ngram_data

    prepare_ngram_data(config_path)
    return {"output_dir": "data/processed/ngram"}


def step_prepare_translation(config_path: str) -> Dict:
    """Prepare translation, IPA, and dictionary data (Task 2.x)."""
    from scripts.prepare_translation_data import prepare_translation_data

    prepare_translation_data(config_path)
    return {"output_dir": "data/processed/translation"}


def step_prepare_tts(config_path: str) -> Dict:
    """Prepare TTS data and speaker embeddings (Task 3.x)."""
    from scripts.prepare_tts_data import prepare_tts_data

    prepare_tts_data(config_path)
    return {"output_dir": "data/processed/tts"}


def step_prepare_antispoof(config_path: str) -> Dict:
    """Prepare anti-spoofing directory structure (Task 4.x scaffold)."""
    from scripts.prepare_antispoof_data import prepare_antispoof_data

    prepare_antispoof_data(config_path)
    return {"output_dir": "data/processed/antispoofing"}


def step_stt(config_path: str) -> Dict:
    """Part I — Denoise + LID bootstrap/train + Constrained Whisper Decoding."""
    from scripts.run_stt_pipeline import run_stt_pipeline

    audio_path = os.environ.get("LECTURE_AUDIO_PATH", "outputs/audio/original_segment.wav")
    if not Path(audio_path).exists():
        raise FileNotFoundError(
            f"Lecture audio not found at {audio_path}. "
            "Place a 10-minute lecture WAV there before running this step."
        )
    device = os.environ.get("TORCH_DEVICE", "").strip() or None
    if device == "mps" or (device is None and _is_mps_available()):
        logger.info("  Forcing CPU for Whisper (MPS lacks float64 support)")
        device = "cpu"

    lid_model_path = "models/lid/best_model.pt"
    if not Path(lid_model_path).exists():
        logger.info("  LID model not found — bootstrapping and training...")
        try:
            from scripts.prepare_lid_data import bootstrap_lid_from_lecture, \
                compute_log_mel_spectrogram, split_data, save_jsonl
            from pathlib import Path as P

            lid_out = P("data/processed/lid")
            lid_out.mkdir(parents=True, exist_ok=True)
            records = bootstrap_lid_from_lecture(audio_path, lid_out)

            if len(records) >= 10:
                train_r, val_r, _ = split_data(records)
                save_jsonl(train_r, lid_out / "audio_train.jsonl")
                save_jsonl(val_r, lid_out / "audio_val.jsonl")

                from scripts.train_lid import train_lid_model
                lid_device = device or "cpu"
                train_lid_model(
                    str(lid_out / "audio_train.jsonl"),
                    str(lid_out / "audio_val.jsonl"),
                    output_dir="models/lid",
                    epochs=15,
                    batch_size=16,
                    device=lid_device,
                )
                logger.info("  LID model trained successfully")
            else:
                logger.warning("  Too few bootstrap chunks (%d); skipping LID training", len(records))
        except Exception as e:
            logger.warning("  LID bootstrap/training failed: %s", e)

    results = run_stt_pipeline(
        audio_path=audio_path,
        config_path=config_path,
        output_dir="outputs",
        whisper_model=os.environ.get("WHISPER_MODEL", "large-v3"),
        ngram_lm_path=os.environ.get("NGRAM_LM_PATH", "data/processed/ngram/maithili_3gram.arpa"),
        technical_vocab_path=os.environ.get("TECHNICAL_VOCAB_PATH", "data/processed/ngram/technical_vocab.json"),
        alpha=float(os.environ.get("NGRAM_ALPHA", "0.3")),
        beta=float(os.environ.get("TECHNICAL_BETA", "3.0")),
        device=device,
    )
    return results


def step_translation(config_path: str) -> Dict:
    """Part II — IPA conversion + Maithili translation."""
    from scripts.run_translation_pipeline import run_translation_pipeline

    transcript_path = "outputs/transcript_code_switched.json"
    if not Path(transcript_path).exists():
        raise FileNotFoundError(
            f"Transcript not found at {transcript_path}. Run the 'stt' step first."
        )
    results = run_translation_pipeline(
        transcript_path=transcript_path,
        config_path=config_path,
        output_dir="outputs",
    )
    return results


def step_voice_cloning(config_path: str) -> Dict:
    """Part III — Speaker embedding + Prosody DTW + TTS synthesis."""
    import yaml
    from scripts.run_voice_cloning_pipeline import run_voice_cloning_pipeline

    student_audio = os.environ.get("STUDENT_VOICE_PATH", "outputs/audio/student_voice_ref.wav")
    if not Path(student_audio).exists():
        raise FileNotFoundError(
            f"Student voice recording not found at {student_audio}. "
            "Record 60 s of your voice and place the WAV file there."
        )

    cfg = {}
    if Path(config_path).exists():
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}

    cfg["student_audio"] = student_audio
    cfg["embedding_method"] = os.environ.get("SPEAKER_EMBEDDING_METHOD", "speechbrain")
    device_env = os.environ.get("TORCH_DEVICE", "").strip()
    if device_env:
        cfg["device"] = device_env

    results = run_voice_cloning_pipeline(config=cfg)
    return results


def step_adversarial(config_path: str) -> Dict:
    """Part IV — Anti-spoofing classifier + FGSM adversarial attack."""
    from scripts.run_adversarial_pipeline import run_adversarial_pipeline

    device = os.environ.get("TORCH_DEVICE", "").strip() or None
    results = run_adversarial_pipeline(
        config_path=config_path,
        epochs=int(os.environ.get("ANTISPOOF_EPOCHS", "50")),
        device=device,
    )
    return results


def step_evaluate(config_path: str) -> Dict:
    """Compute all metrics, generate figures, scaffold IEEE report."""
    from scripts.run_evaluation_pipeline import run_evaluation_pipeline

    results = run_evaluation_pipeline(config_path=config_path)
    return results


# ── step registry (ordered) ────────────────────────────────────────────

STEP_DEFS: List[Dict] = [
    {
        "name": "download",
        "label": "1/10  Download datasets",
        "fn": step_download,
        "group": "Data Acquisition",
    },
    {
        "name": "prepare_lid",
        "label": "2/10  Prepare LID data",
        "fn": step_prepare_lid,
        "group": "Data Preparation",
    },
    {
        "name": "prepare_ngram",
        "label": "3/10  Prepare N-gram LM",
        "fn": step_prepare_ngram,
        "group": "Data Preparation",
    },
    {
        "name": "prepare_translation",
        "label": "4/10  Prepare translation data",
        "fn": step_prepare_translation,
        "group": "Data Preparation",
    },
    {
        "name": "prepare_tts",
        "label": "5/10  Prepare TTS data",
        "fn": step_prepare_tts,
        "group": "Data Preparation",
    },
    {
        "name": "prepare_antispoof",
        "label": "6/10  Prepare anti-spoofing scaffold",
        "fn": step_prepare_antispoof,
        "group": "Data Preparation",
    },
    {
        "name": "stt",
        "label": "7/10  Part I  — STT pipeline",
        "fn": step_stt,
        "group": "Model Pipeline",
    },
    {
        "name": "translation",
        "label": "8/10  Part II — Translation pipeline",
        "fn": step_translation,
        "group": "Model Pipeline",
    },
    {
        "name": "voice_cloning",
        "label": "9/10  Part III — Voice cloning",
        "fn": step_voice_cloning,
        "group": "Model Pipeline",
    },
    {
        "name": "adversarial",
        "label": "10a/10 Part IV — Adversarial + anti-spoofing",
        "fn": step_adversarial,
        "group": "Model Pipeline",
    },
    {
        "name": "evaluate",
        "label": "10b/10 Evaluate + report",
        "fn": step_evaluate,
        "group": "Evaluation",
    },
]

STEP_NAMES = [s["name"] for s in STEP_DEFS]
STEP_MAP = {s["name"]: s for s in STEP_DEFS}


# ── main runner ─────────────────────────────────────────────────────────

def run_pipeline(
    steps: List[str],
    config_path: str,
    resume: bool = False,
    stop_on_error: bool = True,
):
    """Execute pipeline steps with progress tracking and checkpointing."""
    checkpoint = load_checkpoint() if resume else {"completed_steps": [], "results": {}}
    completed = set(checkpoint["completed_steps"])

    steps_to_run = [s for s in steps if s not in completed] if resume else steps
    total = len(steps_to_run)

    if resume and completed:
        logger.info("Resuming from checkpoint — %d step(s) already done: %s",
                     len(completed), ", ".join(sorted(completed)))

    if total == 0:
        logger.info("All requested steps are already completed.")
        return checkpoint["results"]

    pipeline_t0 = time.time()
    logger.info("")
    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║   Speech Understanding PA-2 — Maithili Pipeline         ║")
    logger.info("║   Python + PyTorch | Target LRL: mai_Deva               ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")
    logger.info("")

    for idx, step_name in enumerate(steps_to_run):
        step_def = STEP_MAP.get(step_name)
        if step_def is None:
            logger.error("Unknown step: %s", step_name)
            continue

        progress_bar(idx, total, label=step_def["label"])

        logger.info("")
        logger.info("┌─────────────────────────────────────────────────────────┐")
        logger.info("│  %s", step_def["label"])
        logger.info("└─────────────────────────────────────────────────────────┘")

        t0 = time.time()
        try:
            result = step_def["fn"](config_path)
            elapsed = time.time() - t0

            checkpoint["completed_steps"].append(step_name)
            checkpoint["results"][step_name] = {
                "status": "success",
                "elapsed_sec": round(elapsed, 1),
                "detail": _safe_serialize(result),
            }
            save_checkpoint(checkpoint)

            logger.info("  ✓ Completed in %.1fs", elapsed)

        except FileNotFoundError as e:
            elapsed = time.time() - t0
            checkpoint["results"][step_name] = {
                "status": "blocked",
                "elapsed_sec": round(elapsed, 1),
                "error": str(e),
            }
            save_checkpoint(checkpoint)
            logger.warning("  ⏸ BLOCKED: %s", e)
            if stop_on_error:
                logger.info("  Pipeline paused. Fix the issue and run with --resume.")
                break

        except Exception as e:
            elapsed = time.time() - t0
            checkpoint["results"][step_name] = {
                "status": "failed",
                "elapsed_sec": round(elapsed, 1),
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
            save_checkpoint(checkpoint)
            logger.error("  ✗ FAILED after %.1fs: %s", elapsed, e)
            if stop_on_error:
                logger.info("  Pipeline paused. Fix the issue and run with --resume.")
                break

    progress_bar(total, total, label="Done")

    pipeline_elapsed = time.time() - pipeline_t0

    # ── summary ──
    logger.info("")
    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║  Pipeline Summary                                        ║")
    logger.info("╠══════════════════════════════════════════════════════════╣")

    status_counts = {"success": 0, "failed": 0, "blocked": 0}
    for step_name in steps:
        res = checkpoint["results"].get(step_name, {})
        status = res.get("status", "pending")
        elapsed = res.get("elapsed_sec", 0)

        if status == "success":
            icon = " OK "
            status_counts["success"] += 1
        elif status == "blocked":
            icon = "WAIT"
            status_counts["blocked"] += 1
        elif status == "failed":
            icon = "FAIL"
            status_counts["failed"] += 1
        else:
            icon = " -- "

        label = STEP_MAP.get(step_name, {}).get("label", step_name)
        logger.info("║  [%s] %-40s %6.1fs ║", icon, label, elapsed)

    logger.info("╠══════════════════════════════════════════════════════════╣")
    logger.info(
        "║  Total: %d ok, %d failed, %d blocked  |  %.1fs elapsed      ║",
        status_counts["success"], status_counts["failed"],
        status_counts["blocked"], pipeline_elapsed,
    )
    logger.info("╚══════════════════════════════════════════════════════════╝")
    logger.info("")
    logger.info("Checkpoint saved to %s", CHECKPOINT_PATH)

    return checkpoint["results"]


def _safe_serialize(obj):
    """Convert result to JSON-safe dict, dropping non-serializable values."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        safe = {}
        for k, v in obj.items():
            try:
                json.dumps(v)
                safe[k] = v
            except (TypeError, ValueError):
                safe[k] = str(v)
        return safe
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Speech Understanding PA-2 — Single-command Maithili Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py --all                     Run everything end-to-end
  python pipeline.py --resume                  Resume from last checkpoint
  python pipeline.py --step download           Run one specific step
  python pipeline.py --step prepare_lid stt    Run multiple steps
  python pipeline.py --from stt                Run from 'stt' step onward
  python pipeline.py --list                    Show available steps
  python pipeline.py --reset                   Clear checkpoint and start fresh
""",
    )
    parser.add_argument("--all", action="store_true", help="Run all pipeline steps")
    parser.add_argument("--step", nargs="+", metavar="NAME", help="Run specific step(s)")
    parser.add_argument(
        "--from", dest="from_step", metavar="NAME",
        help="Run from this step through the end",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--config", default=CONFIG_PATH, help="Path to dataset config YAML")
    parser.add_argument("--list", action="store_true", help="List all pipeline steps")
    parser.add_argument("--reset", action="store_true", help="Clear checkpoint file")
    parser.add_argument(
        "--no-stop", action="store_true",
        help="Continue past failures instead of pausing",
    )
    args = parser.parse_args()

    if args.list:
        print("\nAvailable pipeline steps (in dependency order):\n")
        current_group = ""
        checkpoint = load_checkpoint()
        done = set(checkpoint.get("completed_steps", []))
        for i, sd in enumerate(STEP_DEFS, 1):
            if sd["group"] != current_group:
                current_group = sd["group"]
                print(f"  [{current_group}]")
            marker = " ✓" if sd["name"] in done else "  "
            print(f"  {marker} {i:2d}. {sd['name']:25s} — {sd['fn'].__doc__.strip()}")
        print()
        sys.exit(0)

    if args.reset:
        if CHECKPOINT_PATH.exists():
            CHECKPOINT_PATH.unlink()
            print(f"Checkpoint cleared: {CHECKPOINT_PATH}")
        else:
            print("No checkpoint to clear.")
        sys.exit(0)

    if args.all or args.resume:
        steps = STEP_NAMES
    elif args.from_step:
        if args.from_step not in STEP_NAMES:
            print(f"Unknown step: {args.from_step}. Use --list to see options.")
            sys.exit(1)
        start_idx = STEP_NAMES.index(args.from_step)
        steps = STEP_NAMES[start_idx:]
    elif args.step:
        for s in args.step:
            if s not in STEP_NAMES:
                print(f"Unknown step: {s}. Use --list to see options.")
                sys.exit(1)
        steps = args.step
    else:
        parser.print_help()
        sys.exit(1)

    run_pipeline(
        steps=steps,
        config_path=args.config,
        resume=args.resume or args.all,
        stop_on_error=not args.no_stop,
    )


if __name__ == "__main__":
    main()
