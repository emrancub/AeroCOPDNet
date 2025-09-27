# scripts/plot_trainval_panel.py
import argparse, sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ACC_KEYS = [
    ("train_acc", "val_acc"),
    ("accuracy_tr", "accuracy_va"),
    ("acc_tr", "acc_va"),
]
LOSS_KEYS = [
    ("train_loss", "val_loss"),
    ("loss_tr", "loss_va"),
    ("tr_loss", "va_loss"),
]

def _find_cols(df, pairs):
    for tr_key, va_key in pairs:
        if tr_key in df.columns and va_key in df.columns:
            return tr_key, va_key
    # fallback: try to guess
    tr = next((c for c in df.columns if "train" in c and "acc" in c), None)
    va = next((c for c in df.columns if "val" in c and "acc" in c), None)
    if tr and va: return tr, va
    return None, None

def _collect_per_fold(run_dir: Path):
    """Return dict: epoch -> DataFrame with columns from each fold."""
    epochs_dfs = []
    for p in sorted(run_dir.glob("fold_*/*epochs.csv")):
        try:
            df = pd.read_csv(p)
            if "epoch" not in df.columns:
                # try 1-indexed fallback
                df.insert(0, "epoch", np.arange(1, len(df)+1))
            df = df.copy()
            df["fold_id"] = p.parent.name
            epochs_dfs.append(df)
        except Exception as e:
            print(f"[warn] skipping {p}: {e}", file=sys.stderr)
    if not epochs_dfs:
        raise FileNotFoundError(f"No epochs.csv found under {run_dir}")
    return pd.concat(epochs_dfs, ignore_index=True)

def _mean_std_by_epoch(df, key):
    out = df.groupby("epoch")[key].agg(["mean","std"]).reset_index()
    out = out.sort_values("epoch")
    return out["epoch"].values, out["mean"].values, out["std"].fillna(0.0).values

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="CV run dir, e.g. artifacts/cv_old/cvtest_YYYYMMDD_xxx")
    ap.add_argument("--out", default=None, help="Output image path (PNG, PDF, SVG)")
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    df = _collect_per_fold(run_dir)

    # Resolve column names robustly
    tr_acc_key, va_acc_key = _find_cols(df, ACC_KEYS)
    tr_loss_key, va_loss_key = _find_cols(df, LOSS_KEYS)
    if not tr_acc_key or not va_acc_key or not tr_loss_key or not va_loss_key:
        raise ValueError(f"Could not find expected acc/loss columns in {sorted(df.columns)}")

    # === Consistent style (match Mermaid accent) ===
    plt.rcParams.update({
        "figure.figsize": (11, 4.2),
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.edgecolor": "#94A3B8",     # soft gray
        "axes.linewidth": 1.0,
        "xtick.color": "#334155",
        "ytick.color": "#334155",
        "grid.color":  "#E5E7EB",
        "grid.linestyle": "-",
        "grid.linewidth": 0.8,
    })
    ACCENT = "#1D4ED8"   # same as Mermaid stroke
    ACCENT_LT = "#60A5FA" # lighter for train
    NEUTRAL = "#0F172A"  # text

    # === Compute mean±std per epoch ===
    e1, tr_acc_m, tr_acc_s = _mean_std_by_epoch(df, tr_acc_key)
    e2, va_acc_m, va_acc_s = _mean_std_by_epoch(df, va_acc_key)
    e3, tr_loss_m, tr_loss_s = _mean_std_by_epoch(df, tr_loss_key)
    e4, va_loss_m, va_loss_s = _mean_std_by_epoch(df, va_loss_key)

    # align lengths if needed
    E = int(min(len(e1), len(e2), len(e3), len(e4)))
    e1, tr_acc_m, tr_acc_s = e1[:E], tr_acc_m[:E], tr_acc_s[:E]
    e2, va_acc_m, va_acc_s = e2[:E], va_acc_m[:E], va_acc_s[:E]
    e3, tr_loss_m, tr_loss_s = e3[:E], tr_loss_m[:E], tr_loss_s[:E]
    e4, va_loss_m, va_loss_s = e4[:E], va_loss_m[:E], va_loss_s[:E]

    fig = plt.figure()
    # panel A: Accuracy
    ax1 = plt.subplot(1,2,1)
    ax1.set_title("Training vs Validation Accuracy")
    ax1.grid(True, axis="both")
    ax1.plot(e1, tr_acc_m, label="Train", color=ACCENT_LT, linewidth=2.5)
    ax1.fill_between(e1, tr_acc_m - tr_acc_s, tr_acc_m + tr_acc_s, color=ACCENT_LT, alpha=0.20, linewidth=0)
    ax1.plot(e2, va_acc_m, label="Val", color=ACCENT, linewidth=2.5)
    ax1.fill_between(e2, va_acc_m - va_acc_s, va_acc_m + va_acc_s, color=ACCENT, alpha=0.15, linewidth=0)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend(frameon=False)

    # panel B: Loss
    ax2 = plt.subplot(1,2,2)
    ax2.set_title("Training vs Validation Loss")
    ax2.grid(True, axis="both")
    ax2.plot(e3, tr_loss_m, label="Train", color=ACCENT_LT, linewidth=2.5)
    ax2.fill_between(e3, tr_loss_m - tr_loss_s, tr_loss_m + tr_loss_s, color=ACCENT_LT, alpha=0.20, linewidth=0)
    ax2.plot(e4, va_loss_m, label="Val", color=ACCENT, linewidth=2.5)
    ax2.fill_between(e4, va_loss_m - va_loss_s, va_loss_m + va_loss_s, color=ACCENT, alpha=0.15, linewidth=0)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(frameon=False)

    fig.tight_layout()

    out_path = args.out or (run_dir / "fig_trainval_panel.png")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"[saved] {out_path}")

if __name__ == "__main__":
    main()
