import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
)

# ---------- plotting style for papers ----------
def set_pub_style():
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,   # editable text in vector PDFs
        "ps.fonttype": 42,
    })

def best_threshold_by_f1(y_true, y_prob):
    ts = np.linspace(0.05, 0.95, 181)
    f1s = [f1_score(y_true, (y_prob >= t).astype(int)) for t in ts]
    i = int(np.argmax(f1s))
    return float(ts[i]), float(f1s[i])

def draw_confmat(ax, cm_counts, title, labels):
    """cm_counts = [[TN, FP],[FN, TP]] (ints). Annotate counts + row %."""
    im = ax.imshow(cm_counts, aspect="equal")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)

    tn, fp, fn, tp = cm_counts[0,0], cm_counts[0,1], cm_counts[1,0], cm_counts[1,1]
    row_tot = [tn+fp, fn+tp]
    cells = [(0,0,tn,row_tot[0]), (0,1,fp,row_tot[0]), (1,0,fn,row_tot[1]), (1,1,tp,row_tot[1])]
    for i,j,cnt,tot in cells:
        pct = 100.0*cnt/max(tot,1)
        ax.text(j, i, f"{cnt}\n({pct:.1f}%)",
                ha="center", va="center",
                color="white" if pct>60 else "black", fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

def per_fold_summary(run_dir: Path, labels=("Non-COPD","COPD")):
    folds = sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("fold_")])
    assert folds, f"No fold_* dirs in {run_dir}"

    y_true_ref = None
    probs_list = []
    rows = []

    # For ROC/PR overlays
    roc_list, pr_list = [], []

    for fd in folds:
        pred_csv = fd / "test_predictions.csv"
        assert pred_csv.exists(), f"Missing: {pred_csv}"
        dfp = pd.read_csv(pred_csv)
        assert {"y_true","y_prob"}.issubset(dfp.columns)

        y_true = dfp["y_true"].to_numpy().astype(int)
        y_prob = dfp["y_prob"].to_numpy().astype(float)

        if y_true_ref is None:
            y_true_ref = y_true.copy()
        else:
            # Your pipeline keeps test_loader shuffle=False, so order matches.
            # If you ever add a 'path' column you can align by path instead.
            assert np.array_equal(y_true_ref, y_true), "Different order across folds."

        probs_list.append(y_prob)

        # Curves
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        rec, prec, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        roc_list.append((fd.name, fpr, tpr, roc_auc))
        pr_list.append((fd.name, rec, prec, ap))

        # Threshold chosen by F1 (gives “best looking” confusion matrices)
        thr, f1_at_thr = best_threshold_by_f1(y_true, y_prob)
        y_pred = (y_prob >= thr).astype(int)
        cm = confusion_matrix(y_true, y_pred)

        # metrics
        acc = (cm[0,0] + cm[1,1]) / cm.sum()
        prec_s = precision_score(y_true, y_pred, zero_division=0)
        rec_s  = recall_score(y_true, y_pred, zero_division=0)
        spec   = cm[0,0] / max(cm[0,0] + cm[0,1], 1)
        sens   = cm[1,1] / max(cm[1,0] + cm[1,1], 1)

        rows.append({
            "fold": fd.name,
            "N": int(len(y_true)),
            "threshold_F1": thr,
            "Accuracy": acc,
            "F1": f1_at_thr,
            "Precision": prec_s,
            "Recall(Sensitivity)": sens,
            "Specificity": spec,
            "AUC": roc_auc,
            "AP": ap,
            "TN": int(cm[0,0]), "FP": int(cm[0,1]),
            "FN": int(cm[1,0]), "TP": int(cm[1,1]),
        })

        # Save per-fold CM (counts + %)
        fig, ax = plt.subplots(figsize=(6,5))
        draw_confmat(ax, cm.astype(int), title=f"{fd.name} (thr={thr:.2f})", labels=labels)
        out = run_dir / f"fig_confmat_{fd.name}"
        plt.tight_layout(); plt.savefig(out.with_suffix(".png")); plt.savefig(out.with_suffix(".pdf")); plt.close()

    # Ensemble: mean of probabilities across folds
    probs = np.vstack(probs_list)              # (K,N)
    mean_prob = probs.mean(axis=0)
    thr_e, f1_e = best_threshold_by_f1(y_true_ref, mean_prob)
    y_pred_e = (mean_prob >= thr_e).astype(int)
    cm_e = confusion_matrix(y_true_ref, y_pred_e)
    acc_e = (cm_e[0,0] + cm_e[1,1]) / cm_e.sum()

    fig, ax = plt.subplots(figsize=(6,5))
    draw_confmat(ax, cm_e.astype(int), title=f"Ensemble (thr={thr_e:.2f}, Acc={acc_e:.3f})", labels=labels)
    out = run_dir / "fig_confmat_ensemble"
    plt.tight_layout(); plt.savefig(out.with_suffix(".png")); plt.savefig(out.with_suffix(".pdf")); plt.close()

    # Mean of per-fold row-normalized CMs (useful but less “crisp”)
    cms_norm = []
    for r in rows:
        cm = np.array([[r["TN"], r["FP"]],[r["FN"], r["TP"]]], dtype=float)
        cm[0,:] /= max(cm[0,:].sum(), 1.0)
        cm[1,:] /= max(cm[1,:].sum(), 1.0)
        cms_norm.append(cm)
    cm_mean = np.mean(np.stack(cms_norm, axis=0), axis=0)
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm_mean, aspect="equal")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Average Confusion Matrix (mean of row-normalized per-fold)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm_mean[i,j]*100:.1f}%", ha="center", va="center",
                    color="white" if cm_mean[i,j]>0.6 else "black", fontsize=12)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out = run_dir / "fig_confmat_mean_normalized"
    plt.tight_layout(); plt.savefig(out.with_suffix(".png")); plt.savefig(out.with_suffix(".pdf")); plt.close()

    # ROC overlay (per fold + ensemble)
    plt.figure(figsize=(6,5))
    for name, fpr, tpr, roc_auc in roc_list:
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
    fpr_e, tpr_e, _ = roc_curve(y_true_ref, mean_prob)
    auc_e = auc(fpr_e, tpr_e)
    plt.plot(fpr_e, tpr_e, linewidth=3, label=f"Ensemble (AUC={auc_e:.3f})")
    plt.plot([0,1],[0,1], "--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC per Fold + Ensemble"); plt.legend()
    out = run_dir / "fig_roc_allfolds_plus_ensemble"
    plt.tight_layout(); plt.savefig(out.with_suffix(".png")); plt.savefig(out.with_suffix(".pdf")); plt.close()

    # PR overlay (per fold + ensemble)
    plt.figure(figsize=(6,5))
    for name, rec, prec, ap in pr_list:
        plt.plot(rec, prec, label=f"{name} (AP={ap:.3f})")
    prec_e, rec_e, _ = precision_recall_curve(y_true_ref, mean_prob)
    ap_e = average_precision_score(y_true_ref, mean_prob)
    plt.plot(rec_e, prec_e, linewidth=3, label=f"Ensemble (AP={ap_e:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision–Recall per Fold + Ensemble"); plt.legend()
    out = run_dir / "fig_pr_allfolds_plus_ensemble"
    plt.tight_layout(); plt.savefig(out.with_suffix(".png")); plt.savefig(out.with_suffix(".pdf")); plt.close()

    # Per-fold metric bars (optional)
    df_rows = pd.DataFrame(rows)
    for m in ["Accuracy", "F1", "AUC"]:
        plt.figure(figsize=(6,4))
        xs = np.arange(len(df_rows))
        plt.bar(xs, df_rows[m].values)
        plt.xticks(xs, df_rows["fold"].values)
        plt.ylim(0, 1.0)
        plt.ylabel(m); plt.title(f"{m} by fold")
        out = run_dir / f"fig_{m.lower()}_by_fold"
        plt.tight_layout(); plt.savefig(out.with_suffix(".png")); plt.savefig(out.with_suffix(".pdf")); plt.close()

    # Save table with counts + percentages + thresholds
    def add_rates(row):
        neg = row["TN"] + row["FP"]; pos = row["FN"] + row["TP"]
        row["TN%"] = row["TN"]/max(neg,1)
        row["FP%"] = row["FP"]/max(neg,1)
        row["FN%"] = row["FN"]/max(pos,1)
        row["TP%"] = row["TP"]/max(pos,1)
        return row
    df_rows = df_rows.apply(add_rates, axis=1)
    df_rows.to_csv(run_dir / "per_fold_confusion_and_thresholds.csv", index=False)

    # Also report ensemble CM counts+%
    neg_e = int(cm_e[0].sum()); pos_e = int(cm_e[1].sum())
    ens = pd.DataFrame({
        "TN":[int(cm_e[0,0])], "FP":[int(cm_e[0,1])],
        "FN":[int(cm_e[1,0])], "TP":[int(cm_e[1,1])],
        "TN%":[cm_e[0,0]/max(neg_e,1)], "FP%":[cm_e[0,1]/max(neg_e,1)],
        "FN%":[cm_e[1,0]/max(pos_e,1)], "TP%":[cm_e[1,1]/max(pos_e,1)],
        "threshold_F1":[thr_e], "Accuracy":[acc_e],
        "AUC":[roc_auc_score(y_true_ref, mean_prob)],
        "AP":[average_precision_score(y_true_ref, mean_prob)],
        "F1":[f1_e]
    })
    ens.to_csv(run_dir / "ensemble_confusion_counts_and_rates.csv", index=False)

    return df_rows, ens

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="e.g. artifacts/cv/cvtest_20250908_182400_5f")
    ap.add_argument("--labels", default="Non-COPD,COPD")
    args = ap.parse_args()

    set_pub_style()
    run_dir = Path(args.run_dir)
    labels = tuple([s.strip() for s in args.labels.split(",")])
    assert len(labels) == 2

    df_folds, df_ens = per_fold_summary(run_dir, labels=labels)

    print("\n=== Per-fold confusion (counts & percentages) ===")
    print(df_folds[["fold","TN","FP","FN","TP","TN%","FP%","FN%","TP%","threshold_F1","Accuracy","F1","AUC","AP"]])
    print("\n=== Ensemble confusion (counts & percentages) ===")
    print(df_ens)

if __name__ == "__main__":
    main()
