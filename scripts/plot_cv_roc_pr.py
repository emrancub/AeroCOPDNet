# scripts/plot_cv_roc_pr.py
import argparse, glob, numpy as np, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv_dir", required=True, help="dir created by cv_train.py")
    ap.add_argument("--out_prefix", required=True, help="prefix for figures")
    args = ap.parse_args()

    cv_dir = Path(args.cv_dir)
    pred_files = sorted(cv_dir.glob("fold*_preds.csv"))
    assert pred_files, f"No fold*_preds.csv found in {cv_dir}"

    aucs, auprs = [], []
    plt.figure(figsize=(10,5))
    # ROC
    for pf in pred_files:
        df = pd.read_csv(pf)
        y, p = df["label"].values, df["prob"].values
        from sklearn.metrics import roc_curve, precision_recall_curve
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p); aucs.append(auc)
        plt.plot(fpr, tpr, alpha=0.35, label=f"{pf.stem} AUC={auc:.3f}")
    plt.plot([0,1],[0,1],"--",lw=1,color="gray")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC per fold"); plt.legend(ncol=2, fontsize=8)
    plt.tight_layout(); plt.savefig(f"{args.out_prefix}_roc.png", dpi=200)

    # PR
    plt.figure(figsize=(10,5))
    for pf in pred_files:
        df = pd.read_csv(pf); y, p = df["label"].values, df["prob"].values
        from sklearn.metrics import precision_recall_curve
        pr, rc, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p); auprs.append(ap)
        plt.plot(rc, pr, alpha=0.35, label=f"{pf.stem} AUPR={ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR per fold")
    plt.legend(ncol=2, fontsize=8); plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_pr.png", dpi=200)

    # Confusion matrices at 0.5:
    import math
    n = len(pred_files); cols = 3; rows = math.ceil(n/cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12,4*rows))
    axes = axes.ravel()
    for ax, pf in zip(axes, pred_files):
        df = pd.read_csv(pf); y, p = df["label"].values, df["prob"].values
        yhat = (p>=0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0,1]).ravel()
        im = ax.imshow([[tn,fp],[fn,tp]], cmap="Blues")
        ax.set_title(pf.stem); ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Pred 0","Pred 1"]); ax.set_yticklabels(["True 0","True 1"])
        for i,(r,c) in enumerate([(0,0),(0,1),(1,0),(1,1)]):
            val=[tn,fp,fn,tp][i]; ax.text(c,r,str(val),ha="center",va="center",color="black")
    for k in range(len(pred_files), len(axes)):
        axes[k].axis("off")
    fig.suptitle("Confusion matrices (thr=0.5)")
    fig.tight_layout(rect=[0,0,1,0.97])
    fig.savefig(f"{args.out_prefix}_confmats.png", dpi=200)

    print(f"ROC mean AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f} | "
          f"PR mean AUPR={np.mean(auprs):.4f}±{np.std(auprs):.4f}")

if __name__ == "__main__":
    main()
