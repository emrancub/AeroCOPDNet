# scripts/plot_cv_curves.py
import argparse, glob, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import (
    roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
)

from src.copd.utils import ensure_dir, device_str
from src.copd.data import AudioBinaryDataset
from src.copd.features import build_feature
from src.copd.old_models import build_model

# ---------- helpers to reproduce CV splits ----------
def _stdcol(df, wanted):
    lower = {c.lower(): c for c in df.columns}
    got = []
    for w in wanted:
        got.append(lower.get(w, lower.get("filepath" if w=="path" else w, None)))
    return got

def _group_labels(df):
    g = df.groupby("patient_id")["label"].mean().round().astype(int)
    return g.index.values, g.values

def _stratified_group_kfold(df, n_splits=5, seed=42):
    if "patient_id" in df.columns:
        groups, g_y = _group_labels(df)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for gi_tr, gi_te in skf.split(groups, g_y):
            gtr = set(groups[gi_tr]); gte = set(groups[gi_te])
            tr_idx = df.index[df["patient_id"].isin(gtr)].to_numpy()
            te_idx = df.index[df["patient_id"].isin(gte)].to_numpy()
            yield tr_idx, te_idx
    else:
        y = df["label"].values
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        X_dummy = np.zeros(len(y))
        for i_tr, i_te in skf.split(X_dummy, y):
            yield i_tr, i_te

# ---------- plotting primitives ----------
def _pub():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 12,
        "axes.labelsize": 12,
        "legend.frameon": False
    })

def _line(ax, x, ys, label, alpha=0.25, color=None):
    # ys is list of arrays from folds; draw mean±std
    A = np.vstack([np.asarray(y) for y in ys])
    m, s = np.nanmean(A, axis=0), np.nanstd(A, axis=0, ddof=1)
    ax.plot(x, m, label=label, color=color)
    ax.fill_between(x, m - s, m + s, alpha=alpha, color=color)

def _per_fold(ax, x, y, label_prefix):
    # y is dict {fold: array}
    for k, arr in y.items():
        ax.plot(x, arr, alpha=0.6, label=f"{label_prefix} (fold{k})")

# ---------- evaluation utilities ----------
@torch.no_grad()
def _forward_collect(model, loader, feature, device):
    model.eval()
    probs, labels = [], []
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            wav, y = batch[0], batch[1]
        else:
            wav, y = batch["wav"], batch["label"]
        X = feature(wav).to(device).float()   # (B,1,F,T)
        logits = model(X).view(-1)
        p = torch.sigmoid(logits).cpu().numpy()
        probs.append(p); labels.append(np.asarray(y))
    return np.concatenate(labels), np.concatenate(probs)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs_dir", default="artifacts/logs/cv_old",
                    help="Folder with epochs_<run>_fold*.csv")
    ap.add_argument("--plots_dir",  default="artifacts/plots/cv_old",
                    help="Where to save figures")
    ap.add_argument("--models_dir", default="artifacts/models/cv_old",
                    help="Folder with <run_tag>_fold*_best.pt")
    ap.add_argument("--csv", required=True, help="pooled CSV used for CV")
    ap.add_argument("--run_tag", required=True,
                    help="e.g., cv_20250904_115757_5f (matches filenames)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="aerocpdnet")
    ap.add_argument("--features", default="mel")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--duration", type=float, default=4.0)
    args = ap.parse_args()

    _pub()
    epochs_dir = Path(args.epochs_dir); ensure_dir(epochs_dir)
    plots_dir  = Path(args.plots_dir);  ensure_dir(plots_dir)
    models_dir = Path(args.models_dir); ensure_dir(models_dir)

    # ---------- ACC & LOSS curves from epoch CSVs ----------
    pattern = str(epochs_dir / f"epochs_{args.run_tag}_fold*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"No CSVs match: {pattern}")

    # Collect per-fold arrays aligned by epoch
    tr_loss, tr_acc, val_loss, val_acc = {}, {}, {}, {}
    for f in files:
        df = pd.read_csv(f)
        fold = int(Path(f).stem.split("fold")[-1])
        tr_loss[fold] = df["train_loss"].to_numpy()
        tr_acc[fold]  = df["train_acc"].to_numpy()
        val_loss[fold]= df["val_loss"].to_numpy()
        val_acc[fold] = df["val_acc"].to_numpy()
    # Epoch axis (assume equal length)
    E = len(next(iter(tr_loss.values())))
    x = np.arange(1, E+1)

    # Per-fold overlays
    fig, ax = plt.subplots(1,2, figsize=(12,4))
    _per_fold(ax[0], x, tr_loss, "train_loss"); _per_fold(ax[0], x, val_loss, "val_loss")
    ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Loss"); ax[0].set_title("Loss per fold"); ax[0].legend(ncol=2)

    _per_fold(ax[1], x, tr_acc, "train_acc");  _per_fold(ax[1], x, val_acc, "val_acc")
    ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Accuracy"); ax[1].set_title("Accuracy per fold"); ax[1].legend(ncol=2)
    plt.tight_layout(); plt.savefig(plots_dir / f"perfold_curves_{args.run_tag}.png"); plt.close()

    # Mean ± std curves
    fig, ax = plt.subplots(1,2, figsize=(12,4))
    _line(ax[0], x, list(tr_loss.values()), "train_loss")
    _line(ax[0], x, list(val_loss.values()), "val_loss")
    ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Loss"); ax[0].set_title("Loss (mean ± std)"); ax[0].legend()

    _line(ax[1], x, list(tr_acc.values()), "train_acc")
    _line(ax[1], x, list(val_acc.values()), "val_acc")
    ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Accuracy"); ax[1].set_title("Accuracy (mean ± std)"); ax[1].legend()
    plt.tight_layout(); plt.savefig(plots_dir / f"meanstd_curves_{args.run_tag}.png"); plt.close()

    print(f"[OK] Saved: {plots_dir / f'perfold_curves_{args.run_tag}.png'}")
    print(f"[OK] Saved: {plots_dir / f'meanstd_curves_{args.run_tag}.png'}")

    # ---------- ROC & PR overlays for all folds ----------
    # Rebuild CV split and evaluate best checkpoints
    df = pd.read_csv(args.csv)
    pcol, lcol, idcol = _stdcol(df, ["path","label","patient_id"])
    if pcol is None or lcol is None:
        raise SystemExit("CSV must have 'path' (or 'filepath') and 'label' (and optional patient_id).")
    df = df.rename(columns={pcol:"path", lcol:"label"})
    if idcol: df = df.rename(columns={idcol:"patient_id"})
    full_ds = AudioBinaryDataset(args.csv, sample_rate=args.sr, duration=args.duration, return_path=False)

    feature = build_feature(args.features)
    device  = device_str()
    model_template = build_model(args.model, in_ch=1, n_classes=1).to(device)

    # storage for micro-average
    all_y, all_p = [], []

    rocs = []  # (fpr, tpr, auc, fold)
    prs  = []  # (rec, prec, ap, fold)

    for fold, (tr_all_idx, te_idx) in enumerate(_stratified_group_kfold(df, n_splits=args.k, seed=args.seed), start=1):
        ckpt = models_dir / f"{args.run_tag}_fold{fold}_best.pt"
        if not ckpt.exists():
            print(f"[WARN] Missing checkpoint for fold{fold}: {ckpt} — skipping ROC/PR for this fold")
            continue

        # load fresh model each time
        model = build_model(args.model, in_ch=1, n_classes=1).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()

        te_loader = DataLoader(Subset(full_ds, te_idx.tolist()), batch_size=args.batch_size, shuffle=False, num_workers=0)
        y_true, y_prob = _forward_collect(model, te_loader, feature, device)
        all_y.append(y_true); all_p.append(y_prob)

        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        rocs.append((fpr, tpr, auc, fold))
        # PR
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        prs.append((rec, prec, ap, fold))

    if rocs:
        # overlay ROC
        plt.figure(figsize=(6,5))
        for fpr, tpr, auc, fold in rocs:
            plt.plot(fpr, tpr, label=f"Fold {fold} (AUC={auc:.3f})", alpha=0.9)
        # micro-avg
        y_all = np.concatenate(all_y); p_all = np.concatenate(all_p)
        fpr_m, tpr_m, _ = roc_curve(y_all, p_all)
        auc_m = roc_auc_score(y_all, p_all)
        plt.plot(fpr_m, tpr_m, linewidth=3, label=f"Micro-avg (AUC={auc_m:.3f})")
        plt.plot([0,1],[0,1],"--",lw=1)
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title("ROC across folds"); plt.legend()
        plt.tight_layout(); plt.savefig(plots_dir / f"roc_allfolds_{args.run_tag}.png"); plt.close()
        print(f"[OK] Saved: {plots_dir / f'roc_allfolds_{args.run_tag}.png'}")

    if prs:
        # overlay PR
        plt.figure(figsize=(6,5))
        for rec, prec, ap, fold in prs:
            plt.plot(rec, prec, label=f"Fold {fold} (AP={ap:.3f})", alpha=0.9)
        # micro-avg
        y_all = np.concatenate(all_y); p_all = np.concatenate(all_p)
        prec_m, rec_m, _ = precision_recall_curve(y_all, p_all)
        ap_m = average_precision_score(y_all, p_all)
        plt.plot(rec_m, prec_m, linewidth=3, label=f"Micro-avg (AP={ap_m:.3f})")
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title("Precision–Recall across folds"); plt.legend()
        plt.tight_layout(); plt.savefig(plots_dir / f"pr_allfolds_{args.run_tag}.png"); plt.close()
        print(f"[OK] Saved: {plots_dir / f'pr_allfolds_{args.run_tag}.png'}")


if __name__ == "__main__":
    main()
