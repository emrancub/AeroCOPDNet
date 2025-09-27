import argparse, glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- helpers ---------------------------------------------------------------
def _read_run(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # normalize column names we might see
    cols = {c.lower(): c for c in df.columns}
    # accepted column aliases
    def pick(*names):
        for n in names:
            if n in cols: return cols[n]
        return None

    rename_map = {}
    if pick("epoch")          and "epoch"          != cols.get("epoch","epoch"):          rename_map[cols["epoch"]] = "epoch"
    if pick("train_loss")     : rename_map[pick("train_loss")] = "train_loss"
    if pick("val_loss")       : rename_map[pick("val_loss","valid_loss","validation_loss")] = "val_loss"
    # avoid picking "val_auc" as accuracy — filter by exact name
    if pick("train_acc","train_accuracy"): rename_map[pick("train_acc","train_accuracy")] = "train_acc"
    if pick("val_acc","val_accuracy")    : rename_map[pick("val_acc","val_accuracy")]     = "val_acc"
    if pick("val_auc","auc","auc_roc")   : rename_map[pick("val_auc","auc","auc_roc")]    = "val_auc"

    df = df.rename(columns=rename_map)
    needed = ["epoch","train_loss","val_loss"]
    for n in needed:
        if n not in df.columns:
            raise RuntimeError(f"{csv_path}: missing required column '{n}'. Have: {list(df.columns)}")
    # optional accuracy columns (don’t fail if absent)
    for n in ["train_acc","val_acc","val_auc"]:
        if n not in df.columns: df[n] = np.nan
    return df.sort_values("epoch").reset_index(drop=True)

def _stack_series(dfs, key):
    # align by min length to keep epochs consistent across runs
    L = min(len(df) for df in dfs)
    arr = np.stack([df[key].to_numpy()[:L] for df in dfs], axis=0)
    epochs = dfs[0]["epoch"].to_numpy()[:L]
    return epochs, arr

def _plot_mean_std(ax, epochs, arr, label, lw=2.5):
    mean = np.nanmean(arr, axis=0)
    std  = np.nanstd(arr,  axis=0)
    ax.plot(epochs, mean, label=label, linewidth=lw)
    ax.fill_between(epochs, mean-std, mean+std, alpha=0.18)

# --- main ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="One or more CSVs (or a glob pattern) with columns: epoch, train_loss, val_loss, [train_acc, val_acc, val_auc]")
    ap.add_argument("--out", default="artifacts/figures/train_val_curves.png")
    ap.add_argument("--title", default="Training & Validation (mean ± std)")
    ap.add_argument("--smooth", type=int, default=0, help="rolling window; 0=off")
    args = ap.parse_args()

    # expand globs
    expanded = []
    for pat in args.runs:
        expanded += glob.glob(pat)
    if not expanded:
        raise SystemExit(f"No files matched: {args.runs}")

    dfs = [_read_run(p) for p in expanded]

    # optional smoothing
    if args.smooth and args.smooth > 1:
        win = args.smooth
        for i,df in enumerate(dfs):
            dfs[i] = df.rolling(win, min_periods=1).mean()

    epochs, tr_loss = _stack_series(dfs, "train_loss")
    _,      val_loss = _stack_series(dfs, "val_loss")

    # accuracy is optional
    has_acc = not np.all(np.isnan(np.stack([d["val_acc"].to_numpy()[:len(epochs)] for d in dfs], axis=0)))
    if has_acc:
        _, tr_acc  = _stack_series(dfs, "train_acc")
        _, val_acc = _stack_series(dfs, "val_acc")
    # auc optional
    has_auc = not np.all(np.isnan(np.stack([d["val_auc"].to_numpy()[:len(epochs)] for d in dfs], axis=0)))

    # --- figure
    plt.rcParams.update({"font.size": 12})
    ncols = 2 if has_acc else 1
    fig, axes = plt.subplots(1, ncols, figsize=(11.5, 4.2), constrained_layout=True)
    if ncols == 1: axes = [axes]

    # panel 1: accuracy (if present)
    if has_acc:
        ax = axes[0]
        _plot_mean_std(ax, epochs, tr_acc,  "Train Acc")
        _plot_mean_std(ax, epochs, val_acc, "Val Acc")
        ax.set_title("Accuracy (mean ± std)")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.0); ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)

    # panel (last): loss + (optionally) AUC on twinx
    ax = axes[-1]
    _plot_mean_std(ax, epochs, tr_loss,  "Train Loss")
    _plot_mean_std(ax, epochs, val_loss, "Val Loss")
    ax.set_title("Loss (mean ± std)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="upper right")

    if has_auc:
        _, val_auc = _stack_series(dfs, "val_auc")
        ax2 = ax.twinx()
        mean_auc = np.nanmean(val_auc, axis=0)
        std_auc  = np.nanstd(val_auc, axis=0)
        ax2.plot(epochs, mean_auc, linestyle="--", label="Val AUC")
        ax2.fill_between(epochs, mean_auc-std_auc, mean_auc+std_auc, alpha=0.15)
        ax2.set_ylabel("AUC")
        ax2.set_ylim(0.0, 1.0)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle(args.title, y=1.02)
    fig.savefig(args.out, dpi=220)
    print(f"[OK] saved {args.out}")

    # overlay per-run plots (useful for 5-fold visual)
    if len(dfs) > 1:
        fig2, (axA, axB) = plt.subplots(1, 2 if has_acc else 1, figsize=(12, 4.2), constrained_layout=True)
        if not isinstance(axA, plt.Axes): axA, axB = axA, axB  # silence linter
        # Accuracy overlay
        if has_acc:
            for i,df in enumerate(dfs, 1):
                axA.plot(df["epoch"], df["val_acc"], label=f"run{i}")
            axA.set_title("Validation Accuracy — runs")
            axA.set_xlabel("Epoch"); axA.set_ylabel("Accuracy"); axA.set_ylim(0,1); axA.grid(True, alpha=0.25)
            axA.legend(frameon=False, ncol=2)
        # Loss overlay
        axL = axB if has_acc else axA
        for i,df in enumerate(dfs, 1):
            axL.plot(df["epoch"], df["val_loss"], label=f"run{i}")
        axL.set_title("Validation Loss — runs")
        axL.set_xlabel("Epoch"); axL.set_ylabel("Loss"); axL.grid(True, alpha=0.25)
        axL.legend(frameon=False, ncol=2)
        out2 = str(Path(args.out).with_name(Path(args.out).stem + "_overlay.png"))
        fig2.savefig(out2, dpi=220)
        print(f"[OK] saved {out2}")

if __name__ == "__main__":
    main()
