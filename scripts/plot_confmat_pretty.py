import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, matthews_corrcoef, precision_recall_curve,
    roc_curve, brier_score_loss
)

# --------------------- Styling ---------------------
def _pub():
    plt.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300,
        "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,  # editable text in Illustrator
        "legend.frameon": False
    })

# --------------------- Metrics helpers ---------------------
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
        j = tpr - fpr
        return float(thr[int(np.argmax(j))])
    return 0.5

def summarize(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    acc = (tn + tp) / cm.sum()
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)          # sensitivity
    spec = tn / max(tn + fp, 1)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob)
    ap   = average_precision_score(y_true, y_prob)
    mcc  = matthews_corrcoef(y_true, y_pred) if cm.sum() else 0.0
    bal  = (rec + spec) / 2.0
    brier= brier_score_loss(y_true, y_prob)

    return {
        "cm": cm.astype(int), "threshold": float(thr),
        "accuracy": acc, "precision": prec, "recall": rec,
        "specificity": spec, "f1": f1, "auc": auc, "ap": ap,
        "mcc": mcc, "balanced_acc": bal, "brier": brier
    }

# --------------------- Drawing ---------------------
def _annot_conf(ax, cm, labels=("Negative","Positive"), cmap="Blues"):
    """cm (2x2 int counts). Annotates counts + row-% in each cell."""
    totals = cm.sum(axis=1, keepdims=True).clip(min=1)
    rowpct = cm / totals
    im = ax.imshow(rowpct, cmap=cmap, vmin=0, vmax=1, aspect="equal")

    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")

    for i in range(2):
        for j in range(2):
            cnt = cm[i, j]
            pct = 100.0 * rowpct[i, j]
            txt = f"{cnt}\n({pct:.1f}%)"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if rowpct[i,j] > 0.6 else "black", fontsize=12)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized rate")

def _metrics_box(ax, S):
    ax.axis("off")
    lines = [
        f"Threshold: {S['threshold']:.2f}",
        f"Accuracy : {S['accuracy']*100:.2f}%",
        f"F1-score : {S['f1']:.3f}",
        f"Precision: {S['precision']:.3f}",
        f"Recall/Sens.: {S['recall']:.3f}",
        f"Specificity: {S['specificity']:.3f}",
        f"Balanced Acc.: {S['balanced_acc']:.3f}",
        f"AUC (ROC): {S['auc']:.3f}",
        f"AP (PR) : {S['ap']:.3f}",
        f"MCC : {S['mcc']:.3f}",
        f"Brier: {S['brier']:.4f}",
    ]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left",
            family="monospace", fontsize=11)

def plot_confmat_figure(y_true, y_prob, labels=("Non-COPD","COPD"),
                        thr="f1", title="Confusion Matrix", out=None):
    """Single, high-quality figure with counts+percentages and a metrics panel."""
    _pub()
    thr = best_threshold(y_true, y_prob, mode=thr)
    S = summarize(y_true, y_prob, thr)

    fig = plt.figure(figsize=(8.2, 5.6))
    gs = GridSpec(1, 2, width_ratios=[1.05, 0.9], wspace=0.25, figure=fig)
    ax_cm = fig.add_subplot(gs[0,0])
    ax_tx = fig.add_subplot(gs[0,1])

    _annot_conf(ax_cm, S["cm"], labels=labels, cmap="Blues")
    _metrics_box(ax_tx, S)

    fig.suptitle(title, y=0.98)
    plt.tight_layout(rect=[0,0,1,0.97])

    if out:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out.with_suffix(".png"))
        plt.savefig(out.with_suffix(".pdf"))
    return S

# --------------------- CLI utilities ---------------------
def _load_preds(csv_path: Path):
    df = pd.read_csv(csv_path)
    assert {"y_true","y_prob"}.issubset(df.columns), f"{csv_path} must have y_true,y_prob"
    return df["y_true"].to_numpy().astype(int), df["y_prob"].to_numpy().astype(float)

def plot_single(csv_file, labels, thr, out):
    y_true, y_prob = _load_preds(Path(csv_file))
    S = plot_confmat_figure(y_true, y_prob, labels=labels, thr=thr,
                            title=f"Confusion Matrix ({Path(csv_file).parent.name})",
                            out=out)
    print(f"[OK] Saved: {out}.png / .pdf")
    for k in ["threshold","accuracy","f1","precision","recall","specificity",
              "balanced_acc","auc","ap","mcc","brier"]:
        print(f"{k:>14s}: {S[k]:.6f}")
    print("Counts (TN FP / FN TP):", S["cm"].ravel())

def plot_cv_folder(run_dir, labels, thr):
    run_dir = Path(run_dir)
    folds = sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("fold_")])
    assert folds, f"No fold_* dirs in {run_dir}"

    # per-fold
    all_probs = []
    y_ref = None
    for f in folds:
        csv = f / "test_predictions.csv"
        y, p = _load_preds(csv)
        if y_ref is None: y_ref = y
        else: assert np.array_equal(y_ref, y), "Mismatched order across folds."
        all_probs.append(p)
        out = run_dir / f"fig_confmat_{f.name}"
        plot_confmat_figure(y, p, labels=labels, thr=thr,
                            title=f"Confusion Matrix — {f.name}", out=out)

    # ensemble
    probs = np.vstack(all_probs).mean(axis=0)
    out = run_dir / "fig_confmat_ensemble"
    plot_confmat_figure(y_ref, probs, labels=labels, thr=thr,
                        title="Confusion Matrix — Ensemble (mean prob across folds)",
                        out=out)
    print("[OK] Wrote per-fold + ensemble confusion matrices to:", run_dir)

# --------------------- main ---------------------
def main():
    ap = argparse.ArgumentParser(
        description="Pretty confusion matrices with counts, percentages, and full metrics panel."
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--csv", help="Path to a test_predictions.csv containing y_true,y_prob.")
    g.add_argument("--run_dir", help="CV run folder with fold_*/test_predictions.csv.")
    ap.add_argument("--labels", default="Non-COPD,COPD",
                    help="Comma-separated class labels (neg,pos).")
    ap.add_argument("--thr", default="f1",
                    help="Operating threshold: float (e.g., 0.5), 'f1', or 'youden'.")
    ap.add_argument("--out", help="Output stem for single-CSV mode (no extension).")
    args = ap.parse_args()

    labels = tuple([s.strip() for s in args.labels.split(",")])
    _pub()

    if args.csv:
        assert args.out, "--out is required when using --csv"
        plot_single(args.csv, labels, args.thr, args.out)
    else:
        plot_cv_folder(args.run_dir, labels, args.thr)

if __name__ == "__main__":
    main()
