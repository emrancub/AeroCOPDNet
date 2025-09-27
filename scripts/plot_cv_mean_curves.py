# -*- coding: utf-8 -*-
"""
Aggregate Train/Val curves from CV folds (epochs.csv) and plot mean ± std.

Usage (Windows example):
python -m scripts.plot_cv_mean_curves --run_dir "artifacts\\cv\\cvtest_20250908_182400_5f" --ema 0.12

This looks for: <run_dir>/fold_*/epochs.csv
Outputs:
  <run_dir>/fig_trainval_panel_meanstd.png
  <run_dir>/fig_trainval_panel_meanstd.pdf
  <run_dir>/agg_mean_std.csv
"""

import argparse, glob, os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------- styling for a "journal" look --------------------
plt.rcParams.update({
    "figure.dpi": 180,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.2,
    "axes.labelsize": 13,
    "axes.titlesize": 18,
    "font.size": 12,
    "legend.frameon": False,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

# -------------------- helpers --------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _ema(x, alpha=0.0):
    """Exponential moving average; alpha in [0,1]. 0 disables smoothing."""
    if alpha <= 0:
        return x.copy()
    y = np.empty_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y

def _pick(df, candidates):
    """Return first column name from `candidates` that exists in df (case-insensitive)."""
    low = {c.lower(): c for c in df.columns}
    for name in candidates:
        c = low.get(name.lower())
        if c is not None:
            return c
    return None

def _read_epochs_csv(path):
    """
    Be tolerant to different column namings.
    Expected columns (choose-first):
      - train acc : ['train_acc', 'train_accuracy', 'acc', 'tr_acc']
      - val acc   : ['val_acc', 'val_accuracy', 'va_acc']
      - train loss: ['train_loss', 'tr_loss', 'loss']
      - val loss  : ['val_loss', 'va_loss']
      - epoch     : ['epoch', 'epochs']
    """
    df = pd.read_csv(path)
    e_col = _pick(df, ["epoch", "epochs"])
    if e_col is None:
        # create an epoch axis starting from 1
        df["epoch"] = np.arange(1, len(df) + 1)
        e_col = "epoch"

    tacc = _pick(df, ["train_acc", "train_accuracy", "acc", "tr_acc"])
    vacc = _pick(df, ["val_acc", "val_accuracy", "va_acc"])
    tloss = _pick(df, ["train_loss", "tr_loss", "loss"])
    vloss = _pick(df, ["val_loss", "va_loss"])

    missing = [k for k,v in dict(tacc=tacc,vacc=vacc,tloss=tloss,vloss=vloss).items() if v is None]
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}. Got: {list(df.columns)}")

    return df[[e_col, tacc, vacc, tloss, vloss]].rename(
        columns={e_col:"epoch", tacc:"train_acc", vacc:"val_acc", tloss:"train_loss", vloss:"val_loss"}
    )

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Folder containing fold_*/epochs.csv")
    ap.add_argument("--ema", type=float, default=0.0, help="EMA smoothing factor in [0,1] (e.g., 0.10–0.20)")
    ap.add_argument("--out_name", default="fig_trainval_panel_meanstd", help="Base filename for outputs")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    files = sorted(glob.glob(str(run_dir / "fold_*" / "epochs.csv")))
    if not files:
        raise SystemExit(f"No epochs.csv found under: {run_dir}/fold_*/")

    dfs = []
    for f in files:
        try:
            df = _read_epochs_csv(f)
            dfs.append(df)
        except Exception as e:
            raise SystemExit(f"Failed to read {f}: {e}")

    # Align by the minimum number of epochs across folds
    E = min(len(df) for df in dfs)
    dfs = [df.iloc[:E].reset_index(drop=True) for df in dfs]
    epoch = dfs[0]["epoch"].values

    # Stack arrays: shape (K, E)
    def stack(col):
        A = np.vstack([d[col].values for d in dfs])
        return A

    tr_acc = stack("train_acc")
    va_acc = stack("val_acc")
    tr_loss = stack("train_loss")
    va_loss = stack("val_loss")

    # Optional EMA smoothing (fold-wise)
    if args.ema > 0:
        for A in (tr_acc, va_acc, tr_loss, va_loss):
            for k in range(A.shape[0]):
                A[k, :] = _ema(A[k, :], alpha=args.ema)

    # Compute mean ± std
    def mean_std(A):
        return A.mean(axis=0), A.std(axis=0, ddof=1)

    m_tr_acc, s_tr_acc = mean_std(tr_acc)
    m_va_acc, s_va_acc = mean_std(va_acc)
    m_tr_loss, s_tr_loss = mean_std(tr_loss)
    m_va_loss, s_va_loss = mean_std(va_loss)

    # Save aggregated CSV (for reproducibility)
    out_csv = run_dir / "agg_mean_std.csv"
    pd.DataFrame({
        "epoch": epoch,
        "train_acc_mean": m_tr_acc, "train_acc_std": s_tr_acc,
        "val_acc_mean":   m_va_acc, "val_acc_std":   s_va_acc,
        "train_loss_mean":m_tr_loss,"train_loss_std":s_tr_loss,
        "val_loss_mean":  m_va_loss,"val_loss_std":  s_va_loss,
    }).to_csv(out_csv, index=False)

    # ---------- Plot ----------
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    # Accuracy
    ax0 = ax[0]
    ax0.plot(epoch, m_tr_acc, lw=3, color="#1f77b4", label="Train")
    ax0.fill_between(epoch, m_tr_acc - s_tr_acc, m_tr_acc + s_tr_acc, color="#1f77b4", alpha=0.20)
    ax0.plot(epoch, m_va_acc, lw=3, color="#ff7f0e", label="Val")
    ax0.fill_between(epoch, m_va_acc - s_va_acc, m_va_acc + s_va_acc, color="#ff7f0e", alpha=0.20)
    ax0.set_title("Training vs Validation Accuracy")
    ax0.set_xlabel("Epoch"); ax0.set_ylabel("Accuracy")
    ax0.set_xlim(epoch[0], epoch[-1])
    ax0.set_ylim(0.0, 1.0)
    ax0.legend(loc="lower right", ncol=1)

    # Loss
    ax1 = ax[1]
    ax1.plot(epoch, m_tr_loss, lw=3, color="#1f77b4", label="Train")
    ax1.fill_between(epoch, m_tr_loss - s_tr_loss, m_tr_loss + s_tr_loss, color="#1f77b4", alpha=0.20)
    ax1.plot(epoch, m_va_loss, lw=3, color="#ff7f0e", label="Val")
    ax1.fill_between(epoch, m_va_loss - s_va_loss, m_va_loss + s_va_loss, color="#ff7f0e", alpha=0.20)
    ax1.set_title("Training vs Validation Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_xlim(epoch[0], epoch[-1])
    ax1.legend(loc="upper right", ncol=1)

    fig.tight_layout()
    out_png = run_dir / f"{args.out_name}.png"
    out_pdf = run_dir / f"{args.out_name}.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Aggregated curves saved:\n - {out_png}\n - {out_pdf}\n - {out_csv}")

if __name__ == "__main__":
    main()
