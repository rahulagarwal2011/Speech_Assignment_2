"""
Master dataset downloader for Speech Understanding PA-2.
Downloads ONLY Maithili-related data from AI4Bharat datasets.

SECURITY: This script ONLY uses the HuggingFace `datasets` library and
`huggingface_hub` for downloads.  No wget, curl, or arbitrary URL fetching.

Usage:
    python scripts/download_datasets.py --all
    python scripts/download_datasets.py --dataset indicvoices_r
    python scripts/download_datasets.py --list
"""

import os
import json
import argparse
from pathlib import Path

import yaml
import logging

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ── Dataset Registry ────────────────────────────────────────────────────
#
# Each entry documents *exactly* how to download only the Maithili portion.
#
# "method" can be:
#   "load_dataset"  – use datasets.load_dataset() with config/split
#   "hf_download"   – use huggingface_hub.hf_hub_download() for specific files
#   "load_and_filter" – load full dataset then filter rows in-memory
# ────────────────────────────────────────────────────────────────────────

DATASET_REGISTRY = {
    # ── BPCC: En-Mai parallel pairs ────────────────────────────────────
    # Maithili data lives in nllb_filtered/eng_Latn-mai_Deva/ (parquet files)
    # and also as nllb_filtered/mai_Deva.tsv.  We download only these.
    "bpcc": {
        "method": "hf_download",
        "repo_id": "ai4bharat/BPCC",
        "files": [
            "nllb_filtered/mai_Deva.tsv",
        ],
        "subdirs": [
            "nllb_filtered/eng_Latn-mai_Deva",
        ],
        "save_dir": "data/raw/bpcc",
        "license": "CC0 / CC-BY-4.0",
        "size_estimate": "~12 MB (Maithili only)",
        "description": "Gold-standard En-Mai parallel pairs for translation (Task 2.2)",
    },

    # ── IndicVoices-R: TTS-quality Maithili speech ─────────────────────
    # Usage: load_dataset("ai4bharat/indicvoices_r", "Maithili")
    # Downloads all splits (train + test) for the Maithili config.
    "indicvoices_r": {
        "method": "load_dataset_all_splits",
        "hf_path": "ai4bharat/indicvoices_r",
        "hf_config": "Maithili",
        "save_dir": "data/raw/indicvoices_r",
        "license": "CC BY 4.0",
        "size_estimate": "~50 GB (Maithili audio + transcripts)",
        "description": "TTS-quality Maithili speech at 48 kHz (Task 3.1, 3.3, 5)",
    },

    # ── Aksharantar: Maithili transliteration ──────────────────────────
    "aksharantar": {
        "method": "load_dataset",
        "hf_path": "ai4bharat/aksharantar",
        "hf_config": "mai",
        "splits": ["train"],
        "save_dir": "data/raw/aksharantar",
        "license": "CC0 1.0",
        "size_estimate": "~100 MB",
        "description": "Devanagari-Latin transliteration pairs for G2P (Task 2.1)",
    },

    # ── Bhasha-Abhijnaanam: text LID ───────────────────────────────────
    # Downloads full dataset then filters for Hindi, Maithili, English only.
    "bhasha_abhijnaanam": {
        "method": "load_and_filter",
        "hf_path": "ai4bharat/Bhasha-Abhijnaanam",
        "hf_config": None,
        "splits": ["train"],
        "filter_languages": ["hi", "mai", "en"],
        "save_dir": "data/raw/bhasha_abhijnaanam",
        "license": "CC0 1.0",
        "size_estimate": "~50 MB (after filtering to 3 languages)",
        "description": "Text LID for Hindi/Maithili/English discrimination (Task 1.1)",
    },

    # ── NLLB-Seed: En-Mai high-quality seed pairs ──────────────────────
    "nllb_seed": {
        "method": "load_dataset",
        "hf_path": "allenai/nllb",
        "hf_config": "eng_Latn-mai_Deva",
        "splits": ["train"],
        "save_dir": "data/raw/nllb_seed",
        "license": "CC BY-SA 4.0",
        "size_estimate": "~10 MB",
        "description": "High-quality human-translated En-Mai seed pairs (Task 2.2 validation)",
    },

    # ── Sangraha: Maithili monolingual text ────────────────────────────
    "sangraha": {
        "method": "load_dataset",
        "hf_path": "ai4bharat/sangraha",
        "hf_config": "maithili",
        "splits": ["train"],
        "save_dir": "data/raw/sangraha",
        "license": "CC BY 4.0",
        "size_estimate": "~1 GB",
        "description": "Maithili monolingual text corpus for N-gram LM (Task 1.2)",
    },

    # ── Kathbath: Hindi ASR (no Maithili available) ────────────────────
    # Kathbath does NOT have Maithili.  We download Hindi for LID training
    # (Hindi vs English vs Maithili classification needs Hindi audio).
    "kathbath_hindi": {
        "method": "load_dataset",
        "hf_path": "ai4bharat/Kathbath",
        "hf_config": "hindi",
        "splits": ["test"],
        "save_dir": "data/raw/kathbath",
        "license": "CC0 1.0",
        "size_estimate": "~2 GB (Hindi test split only)",
        "description": "Hindi ASR audio for LID training — Hindi class (Task 1.1)",
    },

    # ── IndicVoices-ST: speech translation triplets ────────────────────
    "indicvoices_st": {
        "method": "load_dataset",
        "hf_path": "ai4bharat/BhasaAnuvaad",
        "hf_config": "mai_Deva-eng_Latn",
        "splits": ["train"],
        "save_dir": "data/raw/indicvoices_st",
        "license": "CC BY 4.0",
        "size_estimate": "~5 GB",
        "description": "Maithili-English speech translation triplets (Bridge Part I-II)",
    },
}

# Datasets that were in the original prompt but don't have Maithili:
# - Samanantar: only 11 Indic languages, Maithili is NOT one of them.
#   BPCC subsumes Samanantar data and does include Maithili.


def _download_load_dataset(entry: dict, hf_token: str = None) -> bool:
    """Download using datasets.load_dataset() with explicit splits."""
    from datasets import load_dataset

    save_dir = Path(entry["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    ok = True

    for split in entry["splits"]:
        try:
            ds = load_dataset(
                entry["hf_path"],
                entry.get("hf_config"),
                split=split,
                token=hf_token,
            )
            output_path = save_dir / split
            ds.save_to_disk(str(output_path))
            logger.info(f"  Saved {split} ({len(ds):,} rows) -> {output_path}")
        except Exception as e:
            logger.error(f"  Failed {split}: {e}")
            logger.info(f"  Manual: https://huggingface.co/datasets/{entry['hf_path']}")
            ok = False
    return ok


def _download_load_dataset_all_splits(entry: dict, hf_token: str = None) -> bool:
    """Download all splits at once using datasets.load_dataset() without specifying split."""
    from datasets import load_dataset

    save_dir = Path(entry["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        ds = load_dataset(
            entry["hf_path"],
            entry.get("hf_config"),
            token=hf_token,
        )

        for split_name, split_ds in ds.items():
            output_path = save_dir / split_name
            split_ds.save_to_disk(str(output_path))
            logger.info(f"  Saved {split_name} ({len(split_ds):,} rows) -> {output_path}")

        logger.info(f"  All splits: {list(ds.keys())}")
        return True

    except Exception as e:
        logger.error(f"  Failed: {e}")
        logger.info(f"  Manual: https://huggingface.co/datasets/{entry['hf_path']}")
        return False


def _download_load_and_filter(entry: dict, hf_token: str = None) -> bool:
    """Download full dataset, then filter to keep only target languages."""
    from datasets import load_dataset

    save_dir = Path(entry["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    keep_langs = set(entry.get("filter_languages", []))
    ok = True

    for split in entry["splits"]:
        try:
            ds = load_dataset(
                entry["hf_path"],
                entry.get("hf_config"),
                split=split,
                token=hf_token,
            )
            original_len = len(ds)

            lang_col = None
            for candidate in ["language", "lang", "label", "lang_code"]:
                if candidate in ds.column_names:
                    lang_col = candidate
                    break

            if lang_col and keep_langs:
                ds = ds.filter(lambda x: x[lang_col] in keep_langs)
                logger.info(
                    f"  Filtered {original_len:,} -> {len(ds):,} rows "
                    f"(kept {keep_langs})"
                )

            output_path = save_dir / split
            ds.save_to_disk(str(output_path))
            logger.info(f"  Saved {split} ({len(ds):,} rows) -> {output_path}")
        except Exception as e:
            logger.error(f"  Failed {split}: {e}")
            ok = False
    return ok


def _download_hf_files(entry: dict, hf_token: str = None) -> bool:
    """Download specific files/subdirectories from a HuggingFace repo."""
    from huggingface_hub import hf_hub_download, list_repo_tree

    save_dir = Path(entry["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    repo_id = entry["repo_id"]
    ok = True

    for filepath in entry.get("files", []):
        try:
            local = hf_hub_download(
                repo_id=repo_id,
                filename=filepath,
                repo_type="dataset",
                local_dir=str(save_dir),
                token=hf_token,
            )
            logger.info(f"  Downloaded {filepath} -> {local}")
        except Exception as e:
            logger.error(f"  Failed to download {filepath}: {e}")
            ok = False

    for subdir in entry.get("subdirs", []):
        try:
            tree = list_repo_tree(repo_id, path_in_repo=subdir, repo_type="dataset", token=hf_token)
            count = 0
            for item in tree:
                if hasattr(item, "rfilename"):
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=item.rfilename,
                        repo_type="dataset",
                        local_dir=str(save_dir),
                        token=hf_token,
                    )
                    count += 1
            logger.info(f"  Downloaded {count} files from {subdir}/")
        except Exception as e:
            logger.error(f"  Failed to list/download {subdir}: {e}")
            ok = False

    return ok


DOWNLOAD_METHODS = {
    "load_dataset": _download_load_dataset,
    "load_dataset_all_splits": _download_load_dataset_all_splits,
    "load_and_filter": _download_load_and_filter,
    "hf_download": _download_hf_files,
}


def download_dataset(name: str, hf_token: str = None) -> bool:
    """Download a single dataset from the registry."""
    if name not in DATASET_REGISTRY:
        logger.error(f"Unknown dataset: {name}. Use --list to see options.")
        return False

    entry = DATASET_REGISTRY[name]
    method = entry.get("method", "load_dataset")
    fn = DOWNLOAD_METHODS.get(method)
    if fn is None:
        logger.error(f"Unknown download method: {method}")
        return False

    logger.info(f"{'='*60}")
    logger.info(f"Downloading: {name}")
    logger.info(f"  {entry['description']}")
    logger.info(f"  License: {entry.get('license', 'unknown')}")
    logger.info(f"  Estimated size: {entry.get('size_estimate', 'unknown')}")
    logger.info(f"  Method: {method}")
    logger.info(f"{'='*60}")

    return fn(entry, hf_token)


def download_all(hf_token: str = None) -> dict:
    """Download all datasets in the registry."""
    results = {}
    for name in DATASET_REGISTRY:
        results[name] = download_dataset(name, hf_token)

    logger.info(f"\n{'='*60}")
    logger.info("Download Summary (Maithili-only)")
    logger.info(f"{'='*60}")
    for name, ok in results.items():
        status = "OK" if ok else "FAIL"
        desc = DATASET_REGISTRY[name]["description"][:50]
        logger.info(f"  [{status:4s}] {name:25s} {desc}")

    successful = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)
    logger.info(f"\n  Total: {successful}/{len(results)} succeeded, {failed} failed")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Maithili-only data from AI4Bharat datasets"
    )
    parser.add_argument("--dataset", type=str, help="Single dataset to download")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace token")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    args = parser.parse_args()

    hf_token = args.token or os.environ.get("HF_TOKEN")

    if args.list:
        print("\nAvailable datasets (Maithili-only downloads):\n")
        for name, entry in DATASET_REGISTRY.items():
            size = entry.get("size_estimate", "?")
            print(f"  {name:25s} {size:>20s}  {entry['description']}")
        print()
        print("Note: Samanantar was removed — it does not contain Maithili.")
        print("      Kathbath does not have Maithili; we download Hindi for LID.")
    elif args.all:
        download_all(hf_token)
    elif args.dataset:
        download_dataset(args.dataset, hf_token)
    else:
        print("Specify --all, --dataset <name>, or --list")
        print(f"Available: {list(DATASET_REGISTRY.keys())}")