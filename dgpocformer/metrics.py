from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


def compute_metrics(labels, preds, probs, classes: Sequence[str]) -> dict:
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    probs = np.asarray(probs)
    class_ids = list(range(len(classes)))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=class_ids, average=None, zero_division=0
    )
    try:
        macro_auc = roc_auc_score(
            label_binarize(labels, classes=class_ids),
            probs,
            multi_class="ovr",
            average="macro",
        )
    except Exception:
        macro_auc = float("nan")
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "support": support.tolist(),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "macro_auc": float(macro_auc),
    }


def summarize_fold_metrics(fold_results: list[dict]) -> dict:
    keys = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_auc"]
    summary = {}
    for key in keys:
        vals = np.asarray([r[key] for r in fold_results], dtype=float)
        summary[f"mean_{key}"] = float(np.mean(vals))
        summary[f"population_std_{key}"] = float(np.std(vals, ddof=0))
    return summary


def predictions_dataframe(paths: Sequence[str], labels, preds, probs, classes: Sequence[str], fold=None) -> pd.DataFrame:
    data = {
        "path": list(paths),
        "true_label": np.asarray(labels, dtype=int),
        "true_class": [classes[int(i)] for i in labels],
        "pred_label": np.asarray(preds, dtype=int),
        "pred_class": [classes[int(i)] for i in preds],
    }
    if fold is not None:
        data["fold"] = fold
    for i, cls in enumerate(classes):
        data[f"prob_{cls}"] = probs[:, i]
    return pd.DataFrame(data)


def save_classification_report(path: str | Path, labels, preds, classes: Sequence[str]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = classification_report(labels, preds, target_names=list(classes), digits=4, zero_division=0)
    path.write_text(report, encoding="utf-8")


def plot_roc_curves(labels, probs, classes: Sequence[str], output_path: str | Path, title: str = "ROC curves"):
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    class_ids = list(range(len(classes)))
    y_bin = label_binarize(labels, classes=class_ids)
    mean_fpr = np.linspace(0, 1, 500)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_title(title, fontsize=13, fontweight="bold")
    class_aucs, interp_tprs = [], []

    for c, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, c], probs[:, c])
        auc_c = auc(fpr, tpr)
        class_aucs.append(float(auc_c))
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tprs.append(interp_tpr)
        ax.plot(fpr, tpr, lw=2.0, alpha=0.85, label=f"{cls.replace('_', ' ')} (AUC = {auc_c:.4f})")

    mean_tpr = np.mean(interp_tprs, axis=0)
    mean_tpr[-1] = 1.0
    macro_auc = auc(mean_fpr, mean_tpr)
    ax.plot(mean_fpr, mean_tpr, lw=2.5, linestyle="--", label=f"Macro average (AUC = {macro_auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle=":", lw=1.0, alpha=0.45, label="Random (AUC = 0.50)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"macro_auc_pooled": float(macro_auc), "class_auc": class_aucs}


def plot_confusion_matrix(labels, preds, classes: Sequence[str], output_path: str | Path, title: str = "Confusion matrix"):
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(labels, preds, labels=list(range(len(classes))))
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    short_names = [c.replace("_", "\n") for c in classes]
    annot_norm = np.array([
        [f"{cm_norm[i, j] * 100:.1f}%\n({cm[i, j]})" for j in range(len(classes))]
        for i in range(len(classes))
    ])

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_norm,
        annot=annot_norm,
        fmt="",
        cmap="Blues",
        xticklabels=short_names,
        yticklabels=short_names,
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.80},
        ax=ax,
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"confusion_matrix": cm.tolist(), "row_normalized": cm_norm.tolist()}
