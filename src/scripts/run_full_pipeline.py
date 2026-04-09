import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "output"
SENTIMENT_SCRIPT = PROJECT_ROOT / "src" / "scripts" / "model_training_multi.py"
THEME_SCRIPT = PROJECT_ROOT / "src" / "scripts" / "theme_extraction.py"

PREDICTION_FILE = OUTPUT_DIR / "predicted_data_multi.csv"
THEME_FILE = OUTPUT_DIR / "theme_extracted_output.csv"
FINAL_FILE = OUTPUT_DIR / "final_dataset.csv"


def run_step(script_path: Path):
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")
    print(f"Running: {script_path.name}")
    subprocess.run([sys.executable, str(script_path)], check=True)


def merge_outputs():
    print("Merging prediction and theme outputs...")
    preds = pd.read_csv(PREDICTION_FILE)
    themes = pd.read_csv(THEME_FILE)

    merge_keys = []
    if "review_id" in preds.columns and "review_id" in themes.columns:
        merge_keys.append("review_id")
    if "text" in preds.columns and "text" in themes.columns:
        merge_keys.append("text")

    if merge_keys:
        merged = themes.merge(preds, on=merge_keys, how="left", suffixes=("", "_pred"))
    else:
        print("No common key found. Merging by row order.")
        merged = pd.concat(
            [themes.reset_index(drop=True), preds.reset_index(drop=True)],
            axis=1,
        )

    merged.to_csv(FINAL_FILE, index=False)
    print(f"Wrote final dataset: {FINAL_FILE}")
    return merged


def write_summary(df: pd.DataFrame):
    print("Writing summary files...")
    dist_file = OUTPUT_DIR / "final_sentiment_distribution.csv"
    count_df = (
        df["predicted_sentiment"]
        .fillna("MISSING")
        .value_counts()
        .rename_axis("predicted_sentiment")
        .reset_index(name="count")
    )
    count_df.to_csv(dist_file, index=False)

    if "themes" in df.columns:
        theme_counts = (
            df["themes"]
            .fillna("")
            .str.split(",")
            .explode()
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .value_counts()
            .rename_axis("theme")
            .reset_index(name="count")
        )
        theme_counts.to_csv(OUTPUT_DIR / "theme_counts.csv", index=False)

    if "mixed_detected" in df.columns:
        mixed_counts = (
            df["mixed_detected"]
            .astype(str)
            .value_counts()
            .rename_axis("mixed_detected")
            .reset_index(name="count")
        )
        mixed_counts.to_csv(OUTPUT_DIR / "mixed_flag_counts.csv", index=False)

    print("Summary files saved.")


def main(skip_sentiment: bool, skip_theme: bool, skip_merge: bool):
    OUTPUT_DIR.mkdir(exist_ok=True)

    if not skip_sentiment:
        run_step(SENTIMENT_SCRIPT)

    if not skip_theme:
        run_step(THEME_SCRIPT)

    if not skip_merge:
        merged = merge_outputs()
        write_summary(merged)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the full sentiment + theme pipeline and merge results."
    )
    parser.add_argument("--skip-sentiment", action="store_true", help="Skip sentiment prediction step")
    parser.add_argument("--skip-theme", action="store_true", help="Skip theme extraction step")
    parser.add_argument("--skip-merge", action="store_true", help="Skip merge step")
    args = parser.parse_args()

    main(
        skip_sentiment=args.skip_sentiment,
        skip_theme=args.skip_theme,
        skip_merge=args.skip_merge,
    )