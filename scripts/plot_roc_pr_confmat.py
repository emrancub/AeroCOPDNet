import argparse
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with y_true/y_prob or label/prob")
    ap.add_argument("--out", default="artifacts/figures/roc_pr_cm.png")
    ap.add_argument("--title", default="Model evaluation")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    # column aliases
    y = df["y_true"] if "y_true" in df else df["label"]
    p = df["y_prob"] if "y_prob" in df else df["prob"]
    y = y.astype(int).to_numpy()
    p = p.astype(float).to_numpy()
    yhat = df["y_pred"].astype(int).to_numpy() if "y_pred" in df else (p >= 0.5).astype(int)

    fpr, tpr, _ = roc_curve(y, p)
    roc_auc = auc(fpr, tpr)
    pr_p, pr_r, _ = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)
    cm = confusion_matrix(y, yhat, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()

    plt.rcParams.update({"font.size": 12})
    fig = plt.figure(figsize=(12,4.2), constrained_layout=True)
    gs = fig.add_gridspec(1,3)

    # ROC
    ax = fig.add_subplot(gs[0,0])
    ax.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    ax.plot([0,1],[0,1],'--',alpha=0.4)
    ax.set_title("ROC"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.grid(True,alpha=0.25); ax.legend(frameon=False)

    # PR
    ax = fig.add_subplot(gs[0,1])
    ax.plot(pr_r, pr_p, label=f"AP={ap:.3f}")
    ax.set_title("Precision–Recall"); ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.grid(True,alpha=0.25); ax.legend(frameon=False)

    # Confusion matrix (normalized)
    ax = fig.add_subplot(gs[0,2])
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    im = ax.imshow(cmn, vmin=0, vmax=1, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]}\n({cmn[i,j]*100:.1f}%)",
                    ha="center", va="center", fontsize=11, color=("black" if cmn[i,j]<0.6 else "white"))
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Non-COPD","COPD"]); ax.set_yticklabels(["Non-COPD","COPD"])
    ax.set_title("Confusion Matrix"); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(args.title, y=1.03)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220)
    print(f"[OK] saved {args.out}")

if __name__ == "__main__":
    main()
