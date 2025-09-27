# -*- coding: utf-8 -*-
"""
Make publication-quality panels for a single CV run directory produced by cv_train_with_test.py.

Input layout expected (as in your screenshot):
  <RUN_DIR>/
    fold_01/epochs.csv
    fold_01/test_predictions.csv    # preferred (y_true,y_prob)
    fold_01/test_metrics.json       # fallback (may contain y_true,y_prob)
    fold_02/...
    ...
    test_per_fold.csv
    test_summary_mean_std.json
    ...

Outputs (saved to <RUN_DIR>):
  fig_trainval_panel.png   # Accuracy and Loss (mean±std) with the house style
  fig_roc_pr_panel.png     # ROC and PR curves across folds + micro-averages
  (optionally also PDFs if --save_pdf is passed)

Usage:
  python -m scripts.plot_pub_figs --run_dir artifacts/cv/cvtest_20250908_182400_5f --ema 0.85 --save_pdf
"""

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

# --------------------------- Style ---------------------------

def set_pub_style():
    plt.rcParams.update({
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "lines.linewidth": 2.4,
    })

def nice(ax, xlabel=None, ylabel=None, title=None):
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if title:  ax.set_title(title, pad=10)
    ax.tick_params(length=4, width=1.0)

# --------------------------- Utils ---------------------------

def ema_1d(y, beta=0.85):
    """Exponential moving average for nicer curves (does NOT change mean/std bands)."""
    if beta is None or beta <= 0 or beta >= 1:
        return np.asarray(y, float)
    y = np.asarray(y, float)
    out = np.empty_like(y, dtype=float)
    acc = 0.0
    for i, v in enumerate(y):
        acc = beta * acc + (1 - beta) * v
        out[i] = acc
    return out

def read_epochs_per_fold(run_dir: Path):
    """Return dicts of arrays per fold for train/val loss/acc, trimmed to common length."""
    folds = sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name.lower().startswith("fold_")])
    if not folds:
        raise SystemExit(f"No fold_* folders under: {run_dir}")

    tr_loss, tr_acc, va_loss, va_acc = {}, {}, {}, {}
    Lmin = None
    for fd in folds:
        csv = fd / "epochs.csv"
        if not csv.exists():
            print(f"[WARN] Missing: {csv}; skipping this fold")
            continue
        df = pd.read_csv(csv)
        need = ["train_loss", "train_acc", "val_loss", "val_acc"]
        for k in need:
            if k not in df.columns:
                raise SystemExit(f"{csv} is missing column '{k}'")
        # store
        tr_loss[fd.name] = df["train_loss"].to_numpy()
        tr_acc [fd.name] = df["train_acc" ].to_numpy()
        va_loss[fd.name] = df["val_loss"  ].to_numpy()
        va_acc [fd.name] = df["val_acc"  ].to_numpy()
        Lmin = len(df) if Lmin is None else min(Lmin, len(df))

    # trim to common length (defensive)
    for d in (tr_loss, tr_acc, va_loss, va_acc):
        for k in list(d.keys()):
            d[k] = d[k][:Lmin]

    epochs = np.arange(1, Lmin + 1)
    return epochs, tr_loss, tr_acc, va_loss, va_acc

def read_test_preds_for_fold(fold_dir: Path):
    """
    Returns y_true, y_prob arrays for this fold.
    Prefers test_predictions.csv. Falls back to test_metrics.json (if it contains arrays).
    """
    csv = fold_dir / "test_predictions.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        if "y_true" in df.columns and "y_prob" in df.columns:
            y = df["y_true"].to_numpy()
            p = df["y_prob"].to_numpy()
            return y, p

    js = fold_dir / "test_metrics.json"
    if js.exists():
        try:
            d = json.loads(js.read_text())
            if isinstance(d, dict) and "y_true" in d and "y_prob" in d:
                y = np.asarray(d["y_true"], float)
                p = np.asarray(d["y_prob"], float)
                return y, p
        except Exception:
            pass

    print(f"[WARN] No test predictions found for {fold_dir.name} (needed for ROC/PR).")
    return None, None

# --------------------------- Plots ---------------------------

def plot_trainval_panel(run_dir: Path, ema_beta=0.0, save_pdf=False):
    epochs, tr_loss, tr_acc, va_loss, va_acc = read_epochs_per_fold(run_dir)

    # stack folds to mean±std
    def mean_std(dct):
        A = np.vstack([v for v in dct.values()])
        return np.nanmean(A, axis=0), np.nanstd(A, axis=0, ddof=1)

    m_tr_acc, s_tr_acc = mean_std(tr_acc)
    m_va_acc, s_va_acc = mean_std(va_acc)
    m_tr_lo , s_tr_lo  = mean_std(tr_loss)
    m_va_lo , s_va_lo  = mean_std(va_loss)

    # optional EMA (visual only)
    v_tr_acc = ema_1d(m_tr_acc, ema_beta)
    v_va_acc = ema_1d(m_va_acc, ema_beta)
    v_tr_lo  = ema_1d(m_tr_lo , ema_beta)
    v_va_lo  = ema_1d(m_va_lo , ema_beta)

    # panel
    set_pub_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)

    # Accuracy
    ax = axes[0]
    ax.fill_between(epochs, m_tr_acc - s_tr_acc, m_tr_acc + s_tr_acc, alpha=0.15)
    ax.fill_between(epochs, m_va_acc - s_va_acc, m_va_acc + s_va_acc, alpha=0.15)
    ax.plot(epochs, v_tr_acc, label="Train", color="#5AA6F1")
    ax.plot(epochs, v_va_acc, label="Val",   color="#0A51BF")
    nice(ax, xlabel="Epoch", ylabel="Accuracy", title="Training vs Validation Accuracy")
    ax.legend()

    # Loss
    ax = axes[1]
    ax.fill_between(epochs, m_tr_lo - s_tr_lo, m_tr_lo + s_tr_lo, alpha=0.15)
    ax.fill_between(epochs, m_va_lo - s_va_lo, m_va_lo + s_va_lo, alpha=0.15)
    ax.plot(epochs, v_tr_lo, label="Train", color="#5AA6F1")
    ax.plot(epochs, v_va_lo, label="Val",   color="#0A51BF")
    nice(ax, xlabel="Epoch", ylabel="Loss", title="Training vs Validation Loss")
    ax.legend()

    out_png = run_dir / "fig_trainval_panel.png"
    fig.savefig(out_png)
    if save_pdf:
        fig.savefig(run_dir / "fig_trainval_panel.pdf")
    plt.close(fig)
    print(f"[OK] Saved {out_png}")

def plot_roc_pr_panel(run_dir: Path, save_pdf=False):
    # collect per-fold predictions
    folds = sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name.lower().startswith("fold_")])
    curves_roc, curves_pr = [], []
    all_y, all_p = [], []

    for fd in folds:
        y, p = read_test_preds_for_fold(fd)
        if y is None:
            continue
        y = y.astype(int); p = p.astype(float)
        all_y.append(y); all_p.append(p)

        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)

        curves_roc.append((fpr, tpr, auc, fd.name))
        curves_pr .append((rec, prec, ap, fd.name))

    if not curves_roc:
        print("[WARN] No ROC/PR curves could be built (no per-fold predictions). Skipping panel.")
        return

    set_pub_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)

    # ROC
    ax = axes[0]
    for fpr, tpr, auc, tag in curves_roc:
        ax.plot(fpr, tpr, alpha=0.8, label=f"{tag.replace('_',' ').title()} (AUC={auc:.3f})")
    y_all = np.concatenate(all_y); p_all = np.concatenate(all_p)
    fpr_m, tpr_m, _ = roc_curve(y_all, p_all)
    auc_m = roc_auc_score(y_all, p_all)
    ax.plot(fpr_m, tpr_m, color="#0A51BF", linewidth=3.0, label=f"Micro-Avg (AUC={auc_m:.3f})")
    ax.plot([0,1],[0,1],"--", color="#888888", linewidth=1)
    nice(ax, xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Across Folds")
    ax.legend(ncol=1)

    # PR
    ax = axes[1]
    for rec, prec, ap, tag in curves_pr:
        ax.plot(rec, prec, alpha=0.8, label=f"{tag.replace('_',' ').title()} (AP={ap:.3f})")
    prec_m, rec_m, _ = precision_recall_curve(y_all, p_all)
    ap_m = average_precision_score(y_all, p_all)
    ax.plot(rec_m, prec_m, color="#0A51BF", linewidth=3.0, label=f"Micro-Avg (AP={ap_m:.3f})")
    nice(ax, xlabel="Recall", ylabel="Precision", title="Precision–Recall Across Folds")
    ax.legend(ncol=1)

    out_png = run_dir / "fig_roc_pr_panel.png"
    fig.savefig(out_png)
    if save_pdf:
        fig.savefig(run_dir / "fig_roc_pr_panel.pdf")
    plt.close(fig)
    print(f"[OK] Saved {out_png}")

# --------------------------- CLI ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Path to cvtest_* run folder")
    ap.add_argument("--ema", type=float, default=0.0, help="EMA smoothing beta (0=off, 0.85–0.9 looks nice)")
    ap.add_argument("--save_pdf", action="store_true", help="Also save PDF versions")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"run_dir not found: {run_dir}")

    plot_trainval_panel(run_dir, ema_beta=args.ema, save_pdf=args.save_pdf)
    plot_roc_pr_panel(run_dir, save_pdf=args.save_pdf)

if __name__ == "__main__":
    main()
