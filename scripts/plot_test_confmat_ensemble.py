import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# ---------- publication style ----------
def _pub():
    plt.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300,
        "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "legend.frameon": False
    })

def load_test_predictions(run_dir: Path):
    """Returns y_true (N,), list of y_prob per fold [F x N]. Assumes same test order across folds."""
    fold_dirs = sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("fold_")])
    assert fold_dirs, f"No fold_* directories in {run_dir}"
    y_trues, probs = [], []
    for fd in fold_dirs:
        df = pd.read_csv(fd / "test_predictions.csv")
        assert {"y_true","y_prob"} <= set(df.columns), f"{fd}/test_predictions.csv must have y_true,y_prob"
        y_trues.append(df["y_true"].to_numpy().astype(int))
        probs.append(df["y_prob"].to_numpy().astype(float))
    # sanity: all y_trues equal
    y0 = y_trues[0]
    for k, yt in enumerate(y_trues[1:], start=2):
        if not np.array_equal(yt, y0):
            raise RuntimeError(f"Test order/labels differ at fold {k}. Cannot ensemble safely.")
    return y0, np.stack(probs, axis=0)  # (N,), (F,N)

def plot_confmat(counts_norm, counts_abs, labels, title, out_stem):
    _pub()
    fig, ax = plt.subplots(1, 1, figsize=(6.2, 5.4))
    im = ax.imshow(counts_norm, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")

    for i in range(2):
        for j in range(2):
            cnt = int(counts_abs[i, j])
            pct = 100.0 * counts_norm[i, j]
            ax.text(j, i, f"{cnt}\n({pct:.1f}%)",
                    ha="center", va="center",
                    color="white" if counts_norm[i,j] > 0.6 else "black", fontsize=12)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized rate")
    if title:
        ax.set_title(title)
    plt.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_stem.with_suffix(".png"))
    plt.savefig(out_stem.with_suffix(".pdf"))
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="Ensembled test confusion matrix from CV fold predictions.")
    ap.add_argument("--run_dir", required=True, help="e.g. artifacts/cv/cvtest_20250908_182400_5f")
    ap.add_argument("--thr", type=float, default=0.5, help="Fixed decision threshold (default 0.5)")
    ap.add_argument("--labels", default="Non-COPD,COPD", help="Comma-separated class names (neg,pos)")
    ap.add_argument("--out", default=None, help="Output stem for figure (defaults to <run_dir>/fig_cm_test_ensemble)")
    args = ap.parse_args()

    labels = tuple([s.strip() for s in args.labels.split(",")])
    run_dir = Path(args.run_dir)
    out_stem = Path(args.out) if args.out else run_dir / "fig_cm_test_ensemble"

    # 1) load all folds' test predictions
    y_true, P = load_test_predictions(run_dir)   # P: (F,N)
    p_ens = P.mean(axis=0)                        # average prob across folds -> (N,)

    # 2) fixed-threshold decisions on test
    thr = float(args.thr)
    y_hat = (p_ens >= thr).astype(int)

    # 3) confusion matrix (absolute counts)
    cm = confusion_matrix(y_true, y_hat, labels=[0,1])  # [[TN,FP],[FN,TP]]
    # row-normalized for color/percentages
    row_sum = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm / row_sum

    # 4) metrics (all on test)
    acc = accuracy_score(y_true, y_hat)
    prec = precision_score(y_true, y_hat)
    rec  = recall_score(y_true, y_hat)            # sensitivity
    f1   = f1_score(y_true, y_hat)
    spec = cm[0,0] / max(1, cm[0,0] + cm[0,1])
    bal_acc = 0.5 * (rec + spec)

    title = (f"Test Confusion Matrix (cross-fold prob. ensemble, thr={thr:.2f})\n"
             f"Acc={acc:.3f}  F1={f1:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  Spec={spec:.3f}  BalAcc={bal_acc:.3f}")
    plot_confmat(cm_norm, cm, labels, title, out_stem)

    # 5) print numbers (nice for the paper’s results section)
    print("\n=== TEST (Ensembled) Results ===")
    print("Counts [[TN, FP], [FN, TP]]:\n", cm)
    print("Row-normalized:\n", np.array2string(cm_norm, formatter={'float_kind': lambda x: f'{x:0.4f}'}))
    print(f"Acc={acc:.4f}  F1={f1:.4f}  Prec={prec:.4f}  Rec/Sens={rec:.4f}  Spec={spec:.4f}  BalAcc={bal_acc:.4f}")
    print(f"Saved figure: {out_stem}.png / .pdf")

if __name__ == "__main__":
    main()
