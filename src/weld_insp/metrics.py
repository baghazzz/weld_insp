"""Metrics, confusion matrices and a results log shared by every lab."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)

from .config import CLASSES, DEFECT_CLASSES


def classification_report_df(y_true, y_pred, classes=CLASSES) -> pd.DataFrame:
    """Per-class precision / recall / F1 / support, plus macro and accuracy rows."""
    labels = list(range(len(classes)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    df = pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s}, index=classes)
    present = s > 0
    df.loc["macro (present classes)"] = [p[present].mean(), r[present].mean(), f[present].mean(), s.sum()]
    df.loc["accuracy"] = [np.nan, np.nan, accuracy_score(y_true, y_pred), s.sum()]
    return df.round(4)


def summary(y_true, y_pred, classes=CLASSES) -> dict:
    """Headline numbers. Macro-F1 averages only over classes present in y_true,
    so a class missing from a test set (e.g. on GDXray) does not count as F1 = 0."""
    labels = sorted(set(np.asarray(y_true).tolist()))
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "n": len(y_true),
    }
    _, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(classes))),
                                                 zero_division=0)
    for c, ri, fi, si in zip(classes, r, f, s):
        if si > 0:
            out[f"recall_{c}"] = ri
            out[f"f1_{c}"] = fi
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in out.items()}


def to_binary(y, classes=CLASSES) -> np.ndarray:
    """Collapse 4 classes to defect (1) vs no defect (0)."""
    nd = classes.index("ND")
    return (np.asarray(y) != nd).astype(int)


def defect_score(probs: np.ndarray, classes=CLASSES) -> np.ndarray:
    """P(defect) = 1 - P(no defect), from softmax probabilities."""
    return 1.0 - probs[:, classes.index("ND")]


def binary_summary(y_true_bin, score, threshold: float = 0.5) -> dict:
    y_pred = (np.asarray(score) >= threshold).astype(int)
    out = summary(y_true_bin, y_pred, classes=["ND", "DEFECT"])
    try:
        out["roc_auc"] = round(float(roc_auc_score(y_true_bin, score)), 4)
    except ValueError:
        out["roc_auc"] = np.nan
    return out


def plot_confusion(y_true, y_pred, classes=CLASSES, title="", ax=None, normalize=True, save=None):
    import matplotlib.pyplot as plt
    labels = list(range(len(classes)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    show = cm / cm.sum(1, keepdims=True).clip(min=1) if normalize else cm
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(show, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    for i in range(len(classes)):
        for j in range(len(classes)):
            txt = f"{show[i, j]:.2f}\n({cm[i, j]})" if normalize else str(cm[i, j])
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if show[i, j] > (0.6 if normalize else show.max() * 0.6) else "black")
    ax.set_xticks(labels, classes)
    ax.set_yticks(labels, classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)
    if own:
        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=150, bbox_inches="tight")
    return cm


KEY_COLS = ["lab", "model", "seed", "train_split", "eval_set"]


def log_result(results_dir: Path, row: dict, file: str = "all_results.csv") -> pd.DataFrame:
    """Record one result in results/<file>. A row with the same lab, model, seed,
    train_split and eval_set replaces the old one, so re-running a notebook
    (Restart and Run All) updates results instead of duplicating them."""
    f = Path(results_dir) / file
    row = {"timestamp": datetime.now().isoformat(timespec="seconds"), **row}
    df = pd.read_csv(f) if f.exists() else pd.DataFrame()
    new = pd.DataFrame([row])
    if len(df):
        keys = [k for k in KEY_COLS if k in new.columns or k in df.columns]
        a = df.reindex(columns=keys).astype(str)
        b = new.reindex(columns=keys).astype(str).iloc[0]
        df = df[~(a == b).all(axis=1)]
    df = pd.concat([df, new], ignore_index=True)
    df.to_csv(f, index=False)
    return df


__all__ = ["classification_report_df", "summary", "to_binary", "defect_score", "binary_summary",
           "plot_confusion", "log_result", "DEFECT_CLASSES"]
