#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dgpocformer.config import ensure_output_dir, load_config
from dgpocformer.data import HistoPoCDatasetFromDF, build_transforms, fit_macenko_normalizer, official_train_test_split, scan_histopoc
from dgpocformer.metrics import compute_metrics, plot_confusion_matrix, plot_roc_curves, predictions_dataframe, save_classification_report, summarize_fold_metrics
from dgpocformer.model import build_model_from_config
from dgpocformer.train_utils import get_device, load_state_dict_flexible, predict, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate one or more DG-POCFormer checkpoints on the fixed Test split.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", nargs="+", required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    out_dir = Path(args.output_dir) if args.output_dir else ensure_output_dir(cfg) / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    classes = cfg["classes"]

    df = scan_histopoc(cfg["data_dir"], classes)
    df_train, df_test = official_train_test_split(df)
    prep_cfg = cfg.get("preprocessing", {})
    stain_normalizer = fit_macenko_normalizer(
        df_train,
        stain_ref_img=prep_cfg.get("stain_ref_img", ""),
        use_macenko=bool(prep_cfg.get("use_macenko", True)),
    )
    _, eval_tf = build_transforms(int(cfg.get("img_size", 256)))
    test_ds = HistoPoCDatasetFromDF(df_test, eval_tf, stain_normalizer)
    test_loader = DataLoader(
        test_ds,
        batch_size=int(cfg.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
        pin_memory=True,
    )

    fold_results = []
    pooled_probs, pooled_preds, pooled_labels, pooled_paths, pooled_ckpt = [], [], [], [], []

    for ckpt_path in args.checkpoint:
        ckpt_path = Path(ckpt_path)
        model = build_model_from_config(cfg, num_classes=len(classes)).to(device)
        model = load_state_dict_flexible(model, ckpt_path, device)
        probs, preds, labels = predict(model, test_loader, device)
        metrics = compute_metrics(labels, preds, probs, classes)
        metrics["checkpoint"] = str(ckpt_path)
        fold_results.append(metrics)

        name = ckpt_path.stem
        pred_df = predictions_dataframe(df_test["path"].tolist(), labels, preds, probs, classes)
        pred_df["checkpoint"] = str(ckpt_path)
        pred_df.to_csv(out_dir / f"predictions_{name}.csv", index=False)
        save_classification_report(out_dir / f"classification_report_{name}.txt", labels, preds, classes)

        pooled_probs.append(probs)
        pooled_preds.append(preds)
        pooled_labels.append(labels)
        pooled_paths.extend(df_test["path"].tolist())
        pooled_ckpt.extend([str(ckpt_path)] * len(df_test))

        print(f"{ckpt_path.name}: Acc={metrics['accuracy']*100:.2f}%, F1={metrics['macro_f1']*100:.2f}%, AUC={metrics['macro_auc']:.4f}")

    result_payload = {"checkpoint_results": fold_results}

    if len(args.checkpoint) > 1:
        result_payload["summary_population_std"] = summarize_fold_metrics(fold_results)
        pooled_probs = np.concatenate(pooled_probs, axis=0)
        pooled_preds = np.concatenate(pooled_preds, axis=0)
        pooled_labels = np.concatenate(pooled_labels, axis=0)
        pooled_metrics = compute_metrics(pooled_labels, pooled_preds, pooled_probs, classes)
        pooled_df = predictions_dataframe(pooled_paths, pooled_labels, pooled_preds, pooled_probs, classes)
        pooled_df["checkpoint"] = pooled_ckpt
        pooled_df.to_csv(out_dir / "pooled_predictions.csv", index=False)
        result_payload["pooled_metrics"] = pooled_metrics
        result_payload["pooled_roc"] = plot_roc_curves(pooled_labels, pooled_probs, classes, out_dir / "pooled_roc.png")
        result_payload["pooled_confusion_matrix"] = plot_confusion_matrix(pooled_labels, pooled_preds, classes, out_dir / "pooled_confusion_matrix.png")
        print(f"Pooled: Acc={pooled_metrics['accuracy']*100:.2f}%, F1={pooled_metrics['macro_f1']*100:.2f}%, AUC={pooled_metrics['macro_auc']:.4f}")

    save_json(out_dir / "evaluation_results.json", result_payload)
    print(f"Evaluation outputs saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
