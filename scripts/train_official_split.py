#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dgpocformer.config import ensure_output_dir, load_config
from dgpocformer.data import (
    build_datasets_for_fold,
    compute_class_weights,
    fit_macenko_normalizer,
    make_internal_folds,
    make_weighted_loader,
    official_train_test_split,
    scan_histopoc,
)
from dgpocformer.losses import FocalLoss
from dgpocformer.metrics import (
    compute_metrics,
    plot_confusion_matrix,
    plot_roc_curves,
    predictions_dataframe,
    save_classification_report,
    summarize_fold_metrics,
)
from dgpocformer.model import build_model_from_config, count_parameters
from dgpocformer.train_utils import build_scheduler, get_device, predict, run_epoch, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train DG-POCFormer with the official Train/Test split.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--device", type=str, default=None, help="Example: cuda, cuda:0, or cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = ensure_output_dir(cfg)
    device = get_device(args.device)
    set_seed(int(cfg.get("seed", 42)))

    classes = cfg["classes"]
    train_cfg = cfg["training"]
    prep_cfg = cfg.get("preprocessing", {})
    batch_size = int(cfg.get("batch_size", 8))
    img_size = int(cfg.get("img_size", 256))
    num_workers = int(cfg.get("num_workers", 0))
    n_folds = int(cfg.get("n_internal_folds", 5))
    seed = int(cfg.get("seed", 42))

    print("=" * 80)
    print("DG-POCFormer official-split training")
    print(f"Device     : {device}")
    print(f"Data root  : {cfg['data_dir']}")
    print(f"Output dir : {out_dir.resolve()}")
    print("=" * 80)
    save_json(out_dir / "run_config.json", cfg)

    df = scan_histopoc(cfg["data_dir"], classes)
    df_train, df_test = official_train_test_split(df)
    print("Dataset counts:")
    print(df.groupby(["class", "split"]).size().unstack(fill_value=0))
    print(f"Official Train images: {len(df_train)}")
    print(f"Fixed Test images    : {len(df_test)}")

    split_dir = out_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(split_dir / "official_train_partition.csv", index=False)
    df_test.to_csv(split_dir / "fixed_test_partition.csv", index=False)

    folds = make_internal_folds(df_train, n_splits=n_folds, seed=seed)
    fold_meta = []
    for fold, (train_idx, val_idx) in enumerate(folds, 1):
        train_fold = df_train.iloc[train_idx].copy()
        val_fold = df_train.iloc[val_idx].copy()
        train_fold.to_csv(split_dir / f"fold{fold}_train.csv", index=False)
        val_fold.to_csv(split_dir / f"fold{fold}_val.csv", index=False)
        fold_meta.append({
            "fold": fold,
            "train_size": int(len(train_fold)),
            "val_size": int(len(val_fold)),
            "val_class_counts": val_fold["class"].value_counts().to_dict(),
        })
    save_json(split_dir / "fold_summary.json", fold_meta)

    stain_normalizer = fit_macenko_normalizer(
        df_train,
        stain_ref_img=prep_cfg.get("stain_ref_img", ""),
        use_macenko=bool(prep_cfg.get("use_macenko", True)),
    )

    fold_results = []
    fold_histories = []
    pooled_probs, pooled_preds, pooled_labels, pooled_paths, pooled_folds = [], [], [], [], []

    for fold, (train_idx, val_idx) in enumerate(folds, 1):
        set_seed(seed + fold)
        print("\n" + "=" * 80)
        print(f"Internal validation run {fold}/{n_folds}")
        print("Training subset: four internal folds from official Train partition")
        print("Validation subset: remaining internal fold from official Train partition")
        print("Final evaluation: fixed official Test partition")
        print("=" * 80)

        train_ds, val_ds, test_ds = build_datasets_for_fold(
            df_train,
            df_test,
            train_idx,
            val_idx,
            img_size=img_size,
            stain_normalizer=stain_normalizer,
        )
        train_loader = make_weighted_loader(
            train_ds,
            batch_size=batch_size,
            classes=classes,
            decidual_weight_boost=float(train_cfg.get("decidual_weight_boost", 3.0)),
            num_workers=num_workers,
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        model = build_model_from_config(cfg, num_classes=len(classes)).to(device)
        params = count_parameters(model)
        print(f"Inference parameters: {params['inference']:,}")

        class_weights = None
        if bool(train_cfg.get("use_class_weights", True)):
            class_weights = compute_class_weights(
                train_ds.labels,
                classes=classes,
                decidual_weight_boost=float(train_cfg.get("decidual_weight_boost", 3.0)),
                device=device,
            )
        criterion = FocalLoss(
            gamma=float(train_cfg.get("focal_gamma", 2.0)),
            weight=class_weights,
            label_smoothing=float(train_cfg.get("label_smoothing", 0.05)),
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(train_cfg.get("learning_rate", 3e-4)),
            weight_decay=float(train_cfg.get("weight_decay", 0.05)),
        )
        scheduler = build_scheduler(
            optimizer,
            warmup_epochs=int(train_cfg.get("warmup_epochs", 10)),
            total_epochs=int(train_cfg.get("epochs", 80)),
        )

        best_val_acc = 0.0
        best_epoch = 0
        patience_counter = 0
        history = {"fold": fold, "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        checkpoint_path = out_dir / f"best_fold{fold}.pth"

        for epoch in range(1, int(train_cfg.get("epochs", 80)) + 1):
            t0 = time.perf_counter()
            train_loss, train_acc = run_epoch(
                model,
                train_loader,
                optimizer,
                device,
                criterion,
                train=True,
                epoch=epoch,
                total_epochs=int(train_cfg.get("epochs", 80)),
                mixup_alpha=float(train_cfg.get("mixup_alpha", 0.3)),
                cgcr_lambda=float(train_cfg.get("cgcr_lambda", 0.05)),
            )
            val_loss, val_acc = run_epoch(
                model,
                val_loader,
                optimizer,
                device,
                criterion,
                train=False,
                epoch=epoch,
                total_epochs=int(train_cfg.get("epochs", 80)),
                mixup_alpha=0.0,
                cgcr_lambda=0.0,
            )
            scheduler.step()

            history["train_loss"].append(float(train_loss))
            history["val_loss"].append(float(val_loss))
            history["train_acc"].append(float(train_acc))
            history["val_acc"].append(float(val_acc))

            improved = val_acc > best_val_acc
            if improved:
                best_val_acc = float(val_acc)
                best_epoch = int(epoch)
                patience_counter = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": cfg,
                        "classes": classes,
                        "fold": fold,
                        "best_epoch": best_epoch,
                        "best_val_acc": best_val_acc,
                    },
                    checkpoint_path,
                )
            else:
                patience_counter += 1

            marker = " *" if improved else ""
            print(
                f"Epoch {epoch:03d}: train_loss={train_loss:.4f}, train_acc={train_acc*100:.2f}%, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc*100:.2f}%, "
                f"lr={optimizer.param_groups[0]['lr']:.2e}, time={time.perf_counter()-t0:.1f}s{marker}"
            )

            if patience_counter >= int(train_cfg.get("patience", 10)):
                print(f"Early stopping at epoch {epoch}.")
                break

        history["best_epoch"] = best_epoch
        history["best_val_acc"] = best_val_acc
        fold_histories.append(history)
        save_json(out_dir / f"history_fold{fold}.json", history)

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        test_probs, test_preds, test_labels = predict(model, test_loader, device)
        metrics = compute_metrics(test_labels, test_preds, test_probs, classes)
        metrics.update({
            "fold": fold,
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "checkpoint": str(checkpoint_path),
            "inference_parameters": params["inference"],
        })
        fold_results.append(metrics)

        pred_df = predictions_dataframe(df_test["path"].tolist(), test_labels, test_preds, test_probs, classes, fold=fold)
        pred_df.to_csv(out_dir / f"test_predictions_fold{fold}.csv", index=False)
        save_classification_report(out_dir / f"classification_report_fold{fold}.txt", test_labels, test_preds, classes)

        pooled_probs.append(test_probs)
        pooled_preds.append(test_preds)
        pooled_labels.append(test_labels)
        pooled_paths.extend(df_test["path"].tolist())
        pooled_folds.extend([fold] * len(df_test))

        print(
            f"Test fold-derived model {fold}: "
            f"Acc={metrics['accuracy']*100:.2f}%, "
            f"P={metrics['macro_precision']*100:.2f}%, "
            f"R={metrics['macro_recall']*100:.2f}%, "
            f"F1={metrics['macro_f1']*100:.2f}%, "
            f"AUC={metrics['macro_auc']:.4f}"
        )

    pooled_probs = np.concatenate(pooled_probs, axis=0)
    pooled_preds = np.concatenate(pooled_preds, axis=0)
    pooled_labels = np.concatenate(pooled_labels, axis=0)

    summary = summarize_fold_metrics(fold_results)
    pooled_metrics = compute_metrics(pooled_labels, pooled_preds, pooled_probs, classes)
    pooled_pred_df = predictions_dataframe(pooled_paths, pooled_labels, pooled_preds, pooled_probs, classes)
    pooled_pred_df["fold"] = pooled_folds
    pooled_pred_df.to_csv(out_dir / "pooled_test_predictions.csv", index=False)
    np.savez_compressed(
        out_dir / "pooled_test_predictions.npz",
        probs=pooled_probs,
        preds=pooled_preds,
        labels=pooled_labels,
    )

    roc_info = plot_roc_curves(
        pooled_labels,
        pooled_probs,
        classes,
        out_dir / "pooled_test_roc.png",
        title="DG-POCFormer pooled test ROC",
    )
    cm_info = plot_confusion_matrix(
        pooled_labels,
        pooled_preds,
        classes,
        out_dir / "pooled_test_confusion_matrix.png",
        title="DG-POCFormer pooled test confusion matrix",
    )

    save_json(
        out_dir / "results_official_split.json",
        {
            "protocol": "official Train/Test split; five class-stratified internal validation runs from Train only; each selected model evaluated on fixed Test",
            "fold_results": fold_results,
            "summary_population_std": summary,
            "pooled_test_metrics": pooled_metrics,
            "pooled_roc": roc_info,
            "pooled_confusion_matrix": cm_info,
            "fold_histories": fold_histories,
        },
    )

    print("\n" + "=" * 80)
    print("Final test-set summary across five fold-derived models")
    print(f"Accuracy : {summary['mean_accuracy']*100:.2f} ± {summary['population_std_accuracy']*100:.2f}%")
    print(f"Precision: {summary['mean_macro_precision']*100:.2f} ± {summary['population_std_macro_precision']*100:.2f}%")
    print(f"Recall   : {summary['mean_macro_recall']*100:.2f} ± {summary['population_std_macro_recall']*100:.2f}%")
    print(f"F1-score : {summary['mean_macro_f1']*100:.2f} ± {summary['population_std_macro_f1']*100:.2f}%")
    print(f"AUC      : {summary['mean_macro_auc']:.4f} ± {summary['population_std_macro_auc']:.4f}")
    print(f"Outputs saved in: {out_dir.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
