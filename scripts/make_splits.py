#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dgpocformer.config import ensure_output_dir, load_config
from dgpocformer.data import make_internal_folds, official_train_test_split, scan_histopoc
from dgpocformer.train_utils import save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Create official Train/Test and internal validation split CSV files.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    out_dir = ensure_output_dir(cfg)
    split_dir = out_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    classes = cfg["classes"]
    df = scan_histopoc(cfg["data_dir"], classes)
    df_train, df_test = official_train_test_split(df)
    folds = make_internal_folds(
        df_train,
        n_splits=int(cfg.get("n_internal_folds", 5)),
        seed=int(cfg.get("seed", 42)),
    )

    df_train.to_csv(split_dir / "official_train_partition.csv", index=False)
    df_test.to_csv(split_dir / "fixed_test_partition.csv", index=False)

    fold_summary = []
    for fold, (train_idx, val_idx) in enumerate(folds, 1):
        train_fold = df_train.iloc[train_idx].copy()
        val_fold = df_train.iloc[val_idx].copy()
        train_fold.to_csv(split_dir / f"fold{fold}_train.csv", index=False)
        val_fold.to_csv(split_dir / f"fold{fold}_val.csv", index=False)
        fold_summary.append({
            "fold": fold,
            "train_size": int(len(train_fold)),
            "val_size": int(len(val_fold)),
            "train_class_counts": train_fold["class"].value_counts().to_dict(),
            "val_class_counts": val_fold["class"].value_counts().to_dict(),
        })

    save_json(split_dir / "fold_summary.json", fold_summary)
    print(f"Split files saved to: {split_dir.resolve()}")
    print(df.groupby(["class", "split"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
