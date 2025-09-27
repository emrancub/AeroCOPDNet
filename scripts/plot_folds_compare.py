# scripts/plot_folds_compare.py
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    confusion_matrix, f1_score
)

# ---------- IO helpers ----------
def _load_fold_preds(fold_dir: Path):
    """
    Returns y_true (np.ndarray), y_prob (np.ndarray) for this fold.
    Prefers test_predictions.csv; falls back to test_metrics.json.
    """
    csvp = fold_dir / "test_predictions.csv"
    if csvp.exists():
        df = pd.read_csv(csvp)
        y_true = df["y_true"].to_numpy().astype(int)
        y_prob = df["y_prob"].to_numpy().astype(float)
        return y_true, y_prob

    js = json.loads(Path(fold_dir / "test_metrics.json").read_text())
    if "y_true" in js and "y_prob" in js:
        y_true = np.asarray(js["y_true"]).astype(int)
        y_prob = np.asarray(js["y_prob"]).astype(float)
        return y_true, y_prob

    raise FileNotFoundError(f"No predictions found in {fold_dir}")

def _load_fold_metrics(fold_dir: Path):
    js = json.loads(Path(fold_dir / "test_metrics.json").read_text())
    # be robust to key variants
    acc = float(js.get("acc", js.get("accuracy", np.nan)))
    loss = float(js.get("loss", np.nan))
    auc = float(js.get("auc", js.get("auc_roc", np.nan)))
    aupr = float(js.get("aupr", js.get("average_precision", np.nan)))
    return {"acc": acc, "loss": loss, "auc": auc, "aupr": aupr}

# ---------- threshold selection ----------
def pick_threshold(y_true, y_prob, method="0.5"):
    """
    method:
      - a float string like "0.5" (fixed threshold)
      - "f1"   : threshold maximizing F1 on this fold
      - "youden": threshold maximizing TPR-FPR (Youden's J)
    """
    if method.replace(".", "", 1).isdigit():
        return float(method)

    fpr, tpr, thr = roc_curve(y_true, y_prob)
    if method.lower() == "youden":
        j = tpr - fpr
        return float(thr[np.argmax(j)])
    if method.lower() == "f1":
        prec, rec, thr_pr = precision_recall_curve(y_true, y_prob)
        # scikit returns thresholds for all but first pair; align sizes
        thr_grid = np.concatenate(([0.0], thr_pr))
        f1s = 2 * (prec * rec) / np.clip(prec + rec, 1e-9, None)
        return float(thr_grid[np.nanargmax(f1s)])
    # default
    return 0.5

# ---------- plotting utilities ----------
def _pub_style():
    plt.rcParams.update({
        "figure.dpi": 170, "savefig.dpi": 170,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 12, "axes.labelsize": 12, "legend.frameon": False,
    })

def plot_roc_allfolds(fold_data, out_png, out_pdf=None):
    _pub_style()
    # fold_data: dict[fold] -> (y_true, y_prob)
    plt.figure(figsize=(6.6, 5.2))
    fpr_grid = np.linspace(0, 1, 201)
    tprs = []

    for k in sorted(fold_data):
        y, p = fold_data[k]
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        tpr_i = np.interp(fpr_grid, fpr, tpr)
        tpr_i[0] = 0.0
        tprs.append(tpr_i)
        plt.plot(fpr, tpr, alpha=0.9, label=f"Fold {k} (AUC={auc:.3f})")

    # mean ± std band
    m = np.mean(tprs, axis=0); s = np.std(tprs, axis=0, ddof=1)
    plt.plot(fpr_grid, m, linewidth=3, label=f"Mean", alpha=0.9)
    plt.fill_between(fpr_grid, np.clip(m - s, 0, 1), np.clip(m + s, 0, 1), alpha=0.15)

    plt.plot([0, 1], [0, 1], "--", lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC across folds"); plt.legend()
    plt.tight_layout(); plt.savefig(out_png)
    if out_pdf: plt.savefig(out_pdf)
    plt.close()

def plot_pr_allfolds(fold_data, out_png, out_pdf=None):
    _pub_style()
    plt.figure(figsize=(6.6, 5.2))
    rec_grid = np.linspace(0, 1, 201)
    precs = []

    for k in sorted(fold_data):
        y, p = fold_data[k]
        prec, rec, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        # interpolate precision at common recall grid (monotone in rec)
        prec_i = np.interp(rec_grid, rec[::-1], prec[::-1])
        precs.append(prec_i)
        plt.plot(rec, prec, alpha=0.9, label=f"Fold {k} (AP={ap:.3f})")

    m = np.mean(precs, axis=0); s = np.std(precs, axis=0, ddof=1)
    plt.plot(rec_grid, m, linewidth=3, label="Mean", alpha=0.9)
    plt.fill_between(rec_grid, np.clip(m - s, 0, 1), np.clip(m + s, 0, 1), alpha=0.15)

    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision–Recall across folds"); plt.legend()
    plt.tight_layout(); plt.savefig(out_png)
    if out_pdf: plt.savefig(out_pdf)
    plt.close()

def _cm_text(ax, cm, normalize=False):
    fmt = ".2f" if normalize else "d"
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center", fontsize=11)

def plot_confusions(fold_data, thr_str, out_grid_png, out_grid_pdf,
                    out_mean_png, out_mean_pdf):
    """
    Draw a 2xN grid of per-fold confusion matrices (normalized %),
    plus an averaged confusion matrix across folds (element-wise mean of normalized CMs).
    """
    _pub_style()
    folds = sorted(fold_data)
    n = len(folds)
    # --- per fold confusion matrices (normalized) ---
    ncols = min(n, 3)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0*ncols, 3.6*nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])

    cms_norm = []
    for idx, k in enumerate(folds):
        y, p = fold_data[k]
        thr = pick_threshold(y, p, thr_str)
        yhat = (p >= thr).astype(int)
        cm = confusion_matrix(y, yhat, labels=[0,1])  # [[TN,FP],[FN,TP]]
        # normalize by row (true class) so rows sum to 1
        with np.errstate(divide="ignore", invalid="ignore"):
            row_sum = cm.sum(axis=1, keepdims=True).astype(float)
            cmn = np.divide(cm, np.maximum(row_sum, 1), where=row_sum>0)
        cms_norm.append(cmn)

        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        im = ax.imshow(cmn, vmin=0, vmax=1)
        ax.set_title(f"Fold {k} (thr={thr:.2f})")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Negative","Positive"])
        ax.set_yticklabels(["Negative","Positive"])
        _cm_text(ax, cmn, normalize=True)

    # tidy empty axes
    for j in range(n, nrows*ncols):
        r, c = divmod(j, ncols)
        fig.delaxes(axes[r, c])
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, label="Row-normalized rate")
    fig.suptitle("Per-fold Confusion Matrices (row-normalized)", y=0.98)
    plt.tight_layout(rect=[0,0,1,0.95]); plt.savefig(out_grid_png)
    if out_grid_pdf: plt.savefig(out_grid_pdf)
    plt.close()

    # --- averaged confusion matrix (mean of normalized) ---
    mean_cm = np.mean(np.stack(cms_norm, axis=0), axis=0)
    plt.figure(figsize=(4.6, 4.0))
    plt.imshow(mean_cm, vmin=0, vmax=1)
    plt.title("Average Confusion Matrix (mean of normalized per-fold CMs)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.xticks([0,1], ["Negative","Positive"])
    plt.yticks([0,1], ["Negative","Positive"])
    _cm_text(plt.gca(), mean_cm, normalize=True)
    plt.colorbar(shrink=0.85, label="Row-normalized rate")
    plt.tight_layout(); plt.savefig(out_mean_png)
    if out_mean_pdf: plt.savefig(out_mean_pdf)
    plt.close()

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True,
                    help="Path to cv run, e.g. artifacts/cv/cvtest_20250908_182400_5f")
    ap.add_argument("--out_dir", default=None,
                    help="Where to save figures (defaults to run_dir)")
    ap.add_argument("--threshold", default="0.5",
                    help='Decision threshold: "0.5", "f1", or "youden"')
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # gather folds
    fold_dirs = sorted(p for p in run_dir.glob("fold_*") if p.is_dir())
    if not fold_dirs:
        raise SystemExit(f"No fold_* folders in {run_dir}")

    # load predictions + basic metrics
    fold_data = {}
    rows = []
    for fd in fold_dirs:
        fold_id = int(fd.name.split("_")[-1])
        y, p = _load_fold_preds(fd)
        fold_data[fold_id] = (y, p)
        m = _load_fold_metrics(fd)
        # also compute F1 at chosen threshold (for the table)
        thr = pick_threshold(y, p, args.threshold)
        f1 = f1_score(y, (p >= thr).astype(int))
        rows.append({"fold": fold_id, **m, "thr": thr, "f1@thr": f1})

    # save table
    df = pd.DataFrame(rows).sort_values("fold")
    df.to_csv(out_dir / "fold_test_summary.csv", index=False)
    print(df.to_string(index=False))

    # ROC / PR overlays
    plot_roc_allfolds(
        fold_data,
        out_png=out_dir / "fig_roc_allfolds.png",
        out_pdf=out_dir / "fig_roc_allfolds.pdf",
    )
    plot_pr_allfolds(
        fold_data,
        out_png=out_dir / "fig_pr_allfolds.png",
        out_pdf=out_dir / "fig_pr_allfolds.pdf",
    )

    # Confusion matrices (per fold + mean)
    plot_confusions(
        fold_data, args.threshold,
        out_grid_png=out_dir / "fig_cm_perfold.png",
        out_grid_pdf=out_dir / "fig_cm_perfold.pdf",
        out_mean_png=out_dir / "fig_cm_mean.png",
        out_mean_pdf=out_dir / "fig_cm_mean.pdf",
    )
    print(f"\nSaved figures to: {out_dir}")

if __name__ == "__main__":
    main()
