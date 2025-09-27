# -*- coding: utf-8 -*-
"""
train_and_plot_clean_curves.py

Trains AeroCOPDNetLite (or any model from build_model) with strong augmentation
but logs *clean* train & validation metrics (no augmentation, no mixup) using an
EMA model. Produces per-fold epochs.csv and a publication-grade mean±std figure.

Assumes your project provides:
  - src.copd.data.AudioBinaryDataset
  - src.copd.features.build_feature
  - src.copd.models.build_model
and that your CSV has columns: path,label[,patient_id]

Usage example (Windows):
  python -m train_and_plot_clean_curves ^
    --csv artifacts\\splits\\pooled_icbhi_fraiwan_binary.csv ^
    --run_dir artifacts\\cv\\cv_clean_curves_5f ^
    --model aerocpdnetlite2 --features mel ^
    --folds 5 --epochs 100 --batch_size 32 --ema 0.999 --mixup 0.2

"""

import argparse, math, os, json, glob
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# (optional) keep OpenMP deterministic and quiet:
os.environ.setdefault("OMP_NUM_THREADS", "1")

# ===== project imports (adapt if your paths differ) =====
from src.copd.data import AudioBinaryDataset
from src.copd.features import build_feature
from src.copd.models import build_model

# ------------------ helpers: splits ------------------
def _stdcol(df, wanted):
    lower = {c.lower(): c for c in df.columns}
    got = []
    for w in wanted:
        got.append(lower.get(w, lower.get("filepath" if w=="path" else w, None)))
    return got

def _group_labels(df):
    g = df.groupby("patient_id")["label"].mean().round().astype(int)
    return g.index.values, g.values

def stratified_group_kfold(df, n_splits=5, seed=42):
    """Yields (train_idx, val_idx, test_idx) for 5-fold CV with an *external* test partition per fold."""
    # Here we make a single external holdout first (stratified by patient),
    # then do a 5-fold split on the dev pool per fold.
    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
    pcol, lcol, idcol = _stdcol(df, ["path","label","patient_id"])
    if idcol is None:
        # if you lack patient_id, fall back to standard stratified CV
        X_dummy = np.zeros(len(df))
        y = df[lcol].values
        skf_outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr_val, te in skf_outer.split(X_dummy, y):
            # inner split for val from tr_val
            y_dev = y[tr_val]
            skf_inner = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            inner_iter = list(skf_inner.split(np.zeros_like(y_dev), y_dev))
            # rotate inner folds 1-by-1 across outer folds
            k = len(inner_iter)
            fold_id = len(te)  # dummy use to pick idx
            i = (seed % k)
            tr_rel, va_rel = inner_iter[i]
            tr = np.array(tr_val)[tr_rel]
            va = np.array(tr_val)[va_rel]
            yield tr, va, te
        return

    # Patient-wise outer split
    groups, g_y = _group_labels(df)
    skf_outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for gi_dev, gi_te in skf_outer.split(groups, g_y):
        g_dev, g_te = groups[gi_dev], groups[gi_te]
        dev_idx = df.index[df["patient_id"].isin(g_dev)].to_numpy()
        te_idx  = df.index[df["patient_id"].isin(g_te)].to_numpy()

        # Patient-wise inner split for validation
        dev_df = df.loc[dev_idx]
        g_dev2, g_y2 = _group_labels(dev_df)
        skf_inner = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        # pick one inner fold per outer split; rotate with seed
        inner = list(skf_inner.split(g_dev2, g_y2))
        i = (seed % len(inner))
        gi_tr, gi_va = inner[i]
        g_tr, g_va = g_dev2[gi_tr], g_dev2[gi_va]
        tr_idx = dev_df.index[dev_df["patient_id"].isin(g_tr)].to_numpy()
        va_idx = dev_df.index[dev_df["patient_id"].isin(g_va)].to_numpy()
        yield tr_idx, va_idx, te_idx

# ------------------ EMA ------------------
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {k: p.clone().detach() for k, p in model.state_dict().items()}
        self._backup = None
    @torch.no_grad()
    def update(self):
        for k, p in self.model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(p, alpha=1.0 - self.decay)
    def store(self):
        self._backup = {k: p.clone() for k, p in self.model.state_dict().items()}
    @torch.no_grad()
    def copy_to(self):
        for k, p in self.model.state_dict().items():
            p.copy_(self.shadow[k])
    def restore(self):
        if self._backup is not None:
            self.model.load_state_dict(self._backup); self._backup = None

# ------------------ evaluation on CLEAN data ------------------
@torch.no_grad()
def eval_clean(model, loader, feature, device, ema=None, thresh=0.5):
    if ema is not None: ema.store(); ema.copy_to()
    model.eval()
    n = 0; loss_sum = 0.0; correct = 0
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            wav, y = batch[0], batch[1]
        else:
            wav, y = batch["wav"], batch["label"]
        y = torch.as_tensor(y).float().view(-1,1).to(device)
        X = feature(wav).to(device).float()
        logits = model(X)
        loss_sum += F.binary_cross_entropy_with_logits(logits, y).item() * y.size(0)
        pred = (torch.sigmoid(logits) >= thresh).float()
        correct += (pred == y).sum().item()
        n += y.size(0)
    if ema is not None: ema.restore()
    return loss_sum / max(1, n), correct / max(1, n)

# ------------------ training loop ------------------
def mixup_batch(X, y, alpha=0.2):
    if alpha <= 0: return X, y
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(X.size(0), device=X.device)
    X2 = X[idx]; y2 = y[idx]
    return lam*X + (1-lam)*X2, lam*y + (1-lam)*y2

def fit_one_fold(args, fold_id, df, tr_idx, va_idx, te_idx):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Augmented training dataset, clean views for train/val
    ds_train_aug = AudioBinaryDataset(args.csv, sample_rate=args.sr, duration=args.duration,
                                      indices=tr_idx.tolist(), augment=True, return_path=False)
    ds_train_clean = AudioBinaryDataset(args.csv, sample_rate=args.sr, duration=args.duration,
                                        indices=tr_idx.tolist(), augment=False, return_path=False)
    ds_val_clean = AudioBinaryDataset(args.csv, sample_rate=args.sr, duration=args.duration,
                                      indices=va_idx.tolist(), augment=False, return_path=False)

    train_loader = DataLoader(ds_train_aug, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    tr_clean_loader = DataLoader(ds_train_clean, batch_size=args.batch_size*2, shuffle=False, num_workers=4)
    val_loader     = DataLoader(ds_val_clean,   batch_size=args.batch_size*2, shuffle=False, num_workers=4)

    feature = build_feature(args.features)
    model = build_model(args.model, in_ch=1, n_classes=1).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5, min_lr=1e-5)
    ema = EMA(model, decay=args.ema) if args.ema and args.ema < 1.0 else None

    # where to write
    fold_dir = Path(args.run_dir) / f"fold_{fold_id:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    csv_path = fold_dir / "epochs.csv"

    hist = []
    best_val = float("inf"); best_state = None

    for ep in range(1, args.epochs+1):
        model.train(); run_loss = 0.0; n_seen = 0
        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                wav, y = batch[0], batch[1]
            else:
                wav, y = batch["wav"], batch["label"]
            y = torch.as_tensor(y).float().view(-1,1).to(device)
            X = feature(wav).to(device).float()

            X_mix, y_soft = mixup_batch(X, y, alpha=args.mixup)
            logits = model(X_mix)
            loss = F.binary_cross_entropy_with_logits(logits, y_soft)
            opt.zero_grad(); loss.backward(); opt.step()
            if ema: ema.update()

            bs = y.size(0); run_loss += loss.item() * bs; n_seen += bs

        # clean metrics with EMA
        tr_loss, tr_acc = eval_clean(model, tr_clean_loader, feature, device, ema=ema, thresh=args.thresh)
        va_loss, va_acc = eval_clean(model, val_loader,     feature, device, ema=ema, thresh=args.thresh)
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(va_loss)

        row = dict(epoch=ep,
                   train_loss_batch=run_loss/max(1,n_seen),
                   train_loss=tr_loss, train_acc=tr_acc,
                   val_loss=va_loss, val_acc=va_acc)
        hist.append(row)
        print(f"[Fold {fold_id} | {ep:03d}] clean-train acc={tr_acc:.3f} loss={tr_loss:.3f} | "
              f"val acc={va_acc:.3f} loss={va_loss:.3f}")

        # keep best (by val_loss)
        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # save epochs.csv + best.pt
    df_hist = pd.DataFrame(hist)
    df_hist.to_csv(csv_path, index=False)
    if best_state is not None:
        torch.save(best_state, fold_dir / "best.pt")

    # store split indices for reproducibility
    with open(fold_dir / "posthoc_trainval.json", "w") as f:
        json.dump({"train_idx": tr_idx.tolist(), "val_idx": va_idx.tolist(), "test_idx": te_idx.tolist()}, f)

    return df_hist

# ------------------ plotting ------------------
def plot_mean_std(run_dir, out_name="fig_trainval_panel", ema=0.12):
    def ema_filter(x, a):
        if a <= 0: return x
        y = np.zeros_like(x, dtype=float); y[0] = x[0]
        for i in range(1, len(x)):
            y[i] = a*x[i] + (1-a)*y[i-1]
        return y

    files = sorted(glob.glob(str(Path(run_dir) / "fold_*" / "epochs.csv")))
    if not files:
        print(f"[WARN] No epochs.csv found under {run_dir}")
        return
    T, VA, TL, VL = [], [], [], []
    for f in files:
        df = pd.read_csv(f)
        T.append(df["train_acc"].to_numpy())
        VA.append(df["val_acc"].to_numpy())
        TL.append(df["train_loss"].to_numpy())
        VL.append(df["val_loss"].to_numpy())

    T = np.vstack(T); VA = np.vstack(VA); TL = np.vstack(TL); VL = np.vstack(VL)
    mT, sT = T.mean(0), T.std(0, ddof=1); mT = ema_filter(mT, ema)
    mV, sV = VA.mean(0), VA.std(0, ddof=1); mV = ema_filter(mV, ema)
    mTL, sTL = TL.mean(0), TL.std(0, ddof=1); mTL = ema_filter(mTL, ema)
    mVL, sVL = VL.mean(0), VL.std(0, ddof=1); mVL = ema_filter(mVL, ema)

    x = np.arange(1, len(mT)+1)

    plt.rcParams.update({
        "figure.dpi": 200, "savefig.dpi": 300,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 13, "axes.labelsize": 13, "legend.frameon": False
    })

    fig, ax = plt.subplots(1,2, figsize=(12,4.2))
    # Accuracy
    ax[0].plot(x, mT, lw=2.5, label="Training Accuracy")
    ax[0].fill_between(x, mT-sT, mT+sT, alpha=0.18)
    ax[0].plot(x, mV, lw=2.5, label="Validation Accuracy")
    ax[0].fill_between(x, mV-sV, mV+sV, alpha=0.18)
    ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Accuracy")
    ax[0].set_ylim(0.6, 1.0)
    ax[0].legend()

    # Loss
    ax[1].plot(x, mTL, lw=2.5, label="Training Loss")
    ax[1].fill_between(x, mTL-sTL, mTL+sTL, alpha=0.18)
    ax[1].plot(x, mVL, lw=2.5, label="Validation Loss")
    ax[1].fill_between(x, mVL-sVL, mVL+sVL, alpha=0.18)
    ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Loss")
    ax[1].legend()

    fig.suptitle("Training vs Validation (mean ± std over folds)", y=1.02)
    plt.tight_layout()
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(run_dir) / f"{out_name}.png", bbox_inches="tight")
    fig.savefig(Path(run_dir) / f"{out_name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {Path(run_dir)/f'{out_name}.png'} and .pdf")

# ------------------ main ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--model", default="aerocpdnetlite2")
    ap.add_argument("--features", default="mel")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--mixup", type=float, default=0.2)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plot_ema", type=float, default=0.12, help="visual smoothing for the figure")
    args = ap.parse_args()

    # Read CSV and standardize columns
    df = pd.read_csv(args.csv)
    pcol, lcol, idcol = _stdcol(df, ["path", "label", "patient_id"])
    if pcol is None or lcol is None:
        raise SystemExit("CSV must contain columns: path,label[,patient_id]")
    df = df.rename(columns={pcol:"path", lcol:"label"})
    if idcol: df = df.rename(columns={idcol:"patient_id"})

    Path(args.run_dir).mkdir(parents=True, exist_ok=True)

    # 5-fold CV with an external test slice per fold (not used here for curves)
    for k, (tr_idx, va_idx, te_idx) in enumerate(stratified_group_kfold(df, n_splits=args.folds, seed=args.seed), start=1):
        print(f"\n=== Fold {k}/{args.folds}  train={len(tr_idx)}  val={len(va_idx)}  test={len(te_idx)} ===")
        fit_one_fold(args, k, df, tr_idx, va_idx, te_idx)

    plot_mean_std(args.run_dir, out_name="fig_trainval_panel_clean", ema=args.plot_ema)

if __name__ == "__main__":
    main()
