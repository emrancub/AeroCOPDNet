import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, roc_curve

# ---------- publication styling ----------
def _pub():
    plt.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300,
        "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,  # editable text in Illustrator
        "legend.frameon": False
    })

# ---------- threshold selection ----------
def best_threshold(y_true, y_prob, mode="f1"):
    if isinstance(mode, (float, int)):
        return float(mode)
    mode = str(mode).lower()
    if mode == "f1":
        ts = np.linspace(0.01, 0.99, 197)
        f1s = [f1_score(y_true, (y_prob >= t).astype(int)) for t in ts]
        return float(ts[int(np.argmax(f1s))])
    if mode in ("youden", "tpr-fpr", "j"):
        fpr, tpr, thr = roc_curve(y_true, y_prob)
        return float(thr[int(np.argmax(tpr - fpr))])
    return 0.5

# ---------- load one fold ----------
def _load_fold(csv_path: Path):
    df = pd.read_csv(csv_path)
    assert {"y_true","y_prob"}.issubset(df.columns), f"{csv_path} must have y_true,y_prob"
    return df["y_true"].to_numpy().astype(int), df["y_prob"].to_numpy().astype(float)

# ---------- main compute ----------
def avg_confmat(run_dir: Path, thr_mode="f1"):
    fold_dirs = sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("fold_")])
    assert fold_dirs, f"No fold_* dirs under {run_dir}"

    macro_mats = []     # row-normalized 2x2 per fold
    micro_counts = np.zeros((2,2), dtype=np.int64)  # summed raw counts

    for fd in fold_dirs:
        y, p = _load_fold(fd / "test_predictions.csv")
        thr = best_threshold(y, p, thr_mode)
        y_hat = (p >= thr).astype(int)

        cm = confusion_matrix(y, y_hat, labels=[0,1])  # [[TN,FP],[FN,TP]]
        # row-normalize for macro averaging
        row_sum = cm.sum(axis=1, keepdims=True).clip(min=1)
        macro_mats.append(cm / row_sum)

        # accumulate raw counts for micro matrix
        micro_counts += cm

    macro_avg = np.mean(np.stack(macro_mats, axis=0), axis=0)      # 2x2, rows sum to ~1
    micro_row_norm = micro_counts / micro_counts.sum(axis=1, keepdims=True).clip(min=1)

    return macro_avg, micro_counts, micro_row_norm

# ---------- plotting ----------
def plot_avg_confmat(macro_avg, micro_counts, labels=("Non-COPD","COPD"), out_stem: Path=None, title=None):
    """One figure: colors = macro-avg row-normalized; text = micro counts + macro %."""
    _pub()
    fig, ax = plt.subplots(1, 1, figsize=(6.2, 5.4))

    im = ax.imshow(macro_avg, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")

    for i in range(2):
        for j in range(2):
            cnt = int(micro_counts[i, j])
            pct = 100.0 * macro_avg[i, j]
            ax.text(j, i, f"{cnt}\n({pct:.1f}%)",
                    ha="center", va="center",
                    color="white" if macro_avg[i,j] > 0.6 else "black", fontsize=12)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized rate (macro average)")

    if title:
        ax.set_title(title)

    plt.tight_layout()
    if out_stem:
        out_stem.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_stem.with_suffix(".png"))
        plt.savefig(out_stem.with_suffix(".pdf"))
    return fig

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Average confusion matrix across CV folds.")
    ap.add_argument("--run_dir", required=True, help="CV run folder (contains fold_*/test_predictions.csv)")
    ap.add_argument("--thr", default="f1", help="Threshold per fold: float, 'f1' (default), or 'youden'")
    ap.add_argument("--labels", default="Non-COPD,COPD", help="Comma-separated labels for classes (neg,pos)")
    ap.add_argument("--out", default=None, help="Output stem (no ext). Defaults to <run_dir>/fig_cm_average")
    args = ap.parse_args()

    labels = tuple([s.strip() for s in args.labels.split(",")])
    run_dir = Path(args.run_dir)
    out_stem = Path(args.out) if args.out else run_dir / "fig_cm_average"

    macro_avg, micro_counts, micro_row_norm = avg_confmat(run_dir, thr_mode=args.thr)

    title = "Average Confusion Matrix\nMacro: mean of row-normalized per fold  |  Text: pooled counts"
    plot_avg_confmat(macro_avg, micro_counts, labels=labels, out_stem=out_stem, title=title)

    # Also print the numbers to console
    print("\n=== Macro-average row-normalized CM (rows ~ 1.00) ===")
    print(np.array2string(macro_avg, formatter={'float_kind':lambda x:f'{x:0.4f}'}))
    print("\n=== Micro-average counts (summed across folds) ===")
    print(micro_counts)
    print("\n=== Micro-average row-normalized CM (rows ~ 1.00) ===")
    print(np.array2string(micro_row_norm, formatter={'float_kind':lambda x:f'{x:0.4f}'}))
    print(f"\nSaved: {out_stem}.png / .pdf")

if __name__ == "__main__":
    main()
