"""Validate dataset v1: schema, ranges, split leakage, balance.

Usage (from backend/): python -m training.validate_dataset
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

try:
    from training.schema import PAIR_COLUMNS
except ImportError:
    from schema import PAIR_COLUMNS


"""Validate dataset v1/v2: schema, ranges, split leakage, balance.

Usage (from backend/):
    python -m training.validate_dataset         # both v1 and v2
    python -m training.validate_dataset --version 1
    python -m training.validate_dataset --version 2
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

try:
    from training.schema import PAIR_COLUMNS, PAIR_COLUMNS_V2
except ImportError:
    from schema import PAIR_COLUMNS, PAIR_COLUMNS_V2


def check(path, columns, min_rate, max_rate, tag):
    import pandas as pd
    df = pd.read_parquet(path)
    errors = []

    missing = [c for c in columns if c not in df.columns]
    if missing:
        errors.append(f"missing columns: {missing}")

    for col in ["skill_score", "experience_score", "education_score", "bert_proxy", "ats_proxy"]:
        if col in df.columns and ((df[col] < 0).any() or (df[col] > 100).any()):
            errors.append(f"{col} out of 0-100 range")

    if set(df["split"].unique()) != {"train", "val", "test"}:
        errors.append(f"bad splits: {sorted(df['split'].unique())}")

    # No resume leakage across splits
    by_split = {s: set(df[df["split"] == s]["resume_id"]) for s in ["train", "val", "test"]}
    if by_split["train"] & by_split["val"] or by_split["train"] & by_split["test"] or by_split["val"] & by_split["test"]:
        errors.append("resume_id leaks across splits")

    rate = float(df["label_match"].mean())
    if not (min_rate <= rate <= max_rate):
        errors.append(f"match_rate {rate:.3f} outside {min_rate}-{max_rate}")

    if df.isna().any().any():
        errors.append("NaNs present")

    print(f"[{tag}] pairs={len(df)} match_rate={rate:.3f}")
    print(f"[{tag}] split_counts:", df["split"].value_counts().to_dict())
    print(f"[{tag}] competence:", df["competence_label"].value_counts().to_dict())
    if "source" in df.columns:
        print(f"[{tag}] by_source:", df.groupby("source")["label_match"].mean().round(4).to_dict())
    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=int, choices=[1, 2], default=0, help="0 = both")
    args = ap.parse_args()
    proc = os.path.join(BACKEND, "data", "processed")
    versions = [args.version] if args.version else [1, 2]
    errors = []
    for v in versions:
        path = os.path.join(proc, f"pairs_v{'1' if v == 1 else '2'}.parquet")
        if not os.path.exists(path):
            print(f"[v{v}] missing {path}, skipping")
            continue
        if v == 1:
            errors += check(path, PAIR_COLUMNS, 0.25, 0.75, "v1")
        else:
            # v2 is intentionally imbalanced: 69% of kaggle resumes have zero
            # tech-ontology skills, so kaggle pairs are ~99% negatives (realistic
            # hard negatives vs tech JDs). Combined rate ~0.17 is trainable with
            # class weights; positives come from synthetic + IT/Engineering slice.
            errors += check(path, PAIR_COLUMNS_V2, 0.10, 0.75, "v2")
    if errors:
        print("FAILED:")
        for e in errors:
            print(" -", e)
        sys.exit(1)
    print("OK: datasets valid")


if __name__ == "__main__":
    main()
