# scripts/ablation_run.py
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoid OpenMP crash on Windows

import math
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_curve,
                             precision_recall_curve, matthews_corrcoef,
                             balanced_accuracy_score, brier_score_loss)
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------- Matplotlib style ----------
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
})

# ---------- Safe helpers ----------
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def gcd_int(a, b):  # py<3.9 compatible
    while b:
        a, b = b, a % b
    return a

def resample_avoid_resampy(y, sr0, sr):
    if sr0 == sr:
        return y
    try:
        from scipy.signal import resample_poly
        g = gcd_int(int(sr0), int(sr))
        up = int(sr // g)
        down = int(sr0 // g)
        return resample_poly(y, up, down)
    except Exception:
        # Simple linear fallback (OK for training & figs)
        n_new = int(round(len(y) * float(sr) / float(sr0)))
        x_old = np.linspace(0.0, 1.0, num=len(y), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_new, endpoint=False)
        return np.interp(x_new, x_old, y).astype(np.float32, copy=False)

# ---------- Audio / features ----------
def load_audio_fixed(path: str, sr: int, duration: float) -> Tuple[np.ndarray, int]:
    import soundfile as sf
    y, sr0 = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    y = resample_avoid_resampy(y, sr0, sr)
    n_tgt = int(sr * duration)
    if len(y) < n_tgt:
        y = np.pad(y, (0, n_tgt - len(y)))
    else:
        y = y[:n_tgt]
    m = np.max(np.abs(y)) + 1e-8
    return (y / m).astype(np.float32), sr

def mel_db(y: np.ndarray, sr: int, n_fft=1024, hop=256, n_mels=128) -> np.ndarray:
    import librosa
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels, power=2.0
    )
    Sdb = librosa.power_to_db(S, ref=np.max)
    return Sdb.astype(np.float32)

# ---------- Augmentations ----------
def wave_augment(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # time shift
    max_shift = int(0.12 * len(y))
    shift = int(rng.integers(-max_shift, max_shift + 1))
    if shift > 0:
        y = np.concatenate([np.zeros(shift, dtype=y.dtype), y[:-shift]])
    elif shift < 0:
        y = np.concatenate([y[-shift:], np.zeros(-shift, dtype=y.dtype)])

    # mild time-stretch via linear interp
    rate = float(rng.uniform(0.97, 1.03))
    n_new = int(round(len(y) / rate))
    x_old = np.linspace(0.0, 1.0, num=len(y), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_new, endpoint=False)
    y = np.interp(x_new, x_old, y).astype(np.float32)
    if len(y) < len(x_old):
        y = np.pad(y, (0, len(x_old) - len(y)))
    else:
        y = y[:len(x_old)]

    # noise + gain
    p_sig = np.mean(y**2) + 1e-12
    p_noise = p_sig / (10.0**(18.0/10.0))
    y = y + rng.normal(0.0, np.sqrt(p_noise), size=y.shape).astype(np.float32)
    y = y * (10.0 ** (rng.uniform(-4, 4)/20.0))
    m = np.max(np.abs(y)) + 1e-8
    return (y / m).astype(np.float32)

def specaug(mel: torch.Tensor, rng: np.random.Generator,
            F=18, T=35, nF=2, nT=2) -> torch.Tensor:
    # mel: (B,1,F,T)
    B, C, Freq, Time = mel.shape
    out = mel.clone()
    vmin = out.amin().item()
    for _ in range(nF):
        f = int(rng.integers(0, F+1))
        f0 = int(rng.integers(0, max(1, Freq - f)))
        out[:, :, f0:f0+f, :] = vmin
    for _ in range(nT):
        tt = int(rng.integers(0, T+1))
        t0 = int(rng.integers(0, max(1, Time - tt)))
        out[:, :, :, t0:t0+tt] = vmin
    return out

def mixup_batch(X: torch.Tensor, y: torch.Tensor, alpha=0.2,
                rng: np.random.Generator = None) -> Tuple[torch.Tensor, torch.Tensor]:
    # X: (B,1,F,T), y: (B,) float {0,1}
    if alpha <= 0:
        return X, y
    if rng is None:
        rng = np.random.default_rng()
    lam = float(np.random.default_rng().beta(alpha, alpha))
    idx = torch.randperm(X.size(0), device=X.device)
    X = lam * X + (1.0 - lam) * X[idx]
    y = lam * y + (1.0 - lam) * y[idx]
    return X, y

# ---------- Dataset ----------
class CsvAudioDS(Dataset):
    def __init__(self, df: pd.DataFrame, sr: int, duration: float, use_waveaug=False, seed=42):
        self.df = df.reset_index(drop=True)
        self.sr = sr
        self.duration = duration
        self.use_waveaug = use_waveaug
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        p = self.df.loc[i, "path"]
        y = float(self.df.loc[i, "label"])
        wav, _ = load_audio_fixed(p, self.sr, self.duration)
        if self.use_waveaug:
            wav = wave_augment(wav, self.rng)
        return wav, np.float32(y), p

# ---------- Model ----------
def build_backbone(name: str, in_ch=1, n_classes=1, dropout=0.3) -> nn.Module:
    # Use your AeroCOPDNet if available; else fallback to a simple CNN.
    try:
        from src.copd.old_models import build_model as _bm
        return _bm(name, in_ch=in_ch, n_classes=n_classes, dropout=dropout)
    except Exception:
        # Minimal CNN fallback (keeps (B,1,F,T) -> (B,1))
        return nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

# ---------- Metrics ----------
def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr=0.5) -> Dict[str, float]:
    y_pred = (y_prob >= thr).astype(int)
    out = {}
    out["accuracy"] = (y_pred == y_true).mean()
    out["precision"] = precision_score(y_true, y_pred, zero_division=0)
    out["recall"] = recall_score(y_true, y_pred, zero_division=0)           # sensitivity
    # specificity:
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    out["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan
    out["f1"] = f1_score(y_true, y_pred, zero_division=0)
    out["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    out["mcc"] = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred)) > 1 else np.nan
    out["brier"] = brier_score_loss(y_true, y_prob)
    try:
        out["auroc"] = roc_auc_score(y_true, y_prob)
    except Exception:
        out["auroc"] = np.nan
    try:
        out["aupr"] = average_precision_score(y_true, y_prob)
    except Exception:
        out["aupr"] = np.nan
    # curves
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
    except Exception:
        fpr = tpr = prec = rec = np.array([])
    out["fpr"] = fpr; out["tpr"] = tpr; out["prec"] = prec; out["rec"] = rec
    return out

# ---------- Train / Eval ----------
@dataclass
class TrainCfg:
    model: str = "aerocpdnet"
    sr: int = 16000
    duration: float = 4.0
    n_mels: int = 128
    lr: float = 5e-4
    wd: float = 1e-4
    dropout: float = 0.3
    epochs: int = 60
    batch_size: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_waveaug: bool = False
    use_specaug: bool = False
    mixup_alpha: float = 0.0
    seed: int = 42

def batch_to_spec(batch_wav: torch.Tensor, sr: int, n_mels: int) -> torch.Tensor:
    # batch_wav: (B, L)
    specs = []
    for w in batch_wav.cpu().numpy():
        S = mel_db(w, sr=sr, n_mels=n_mels)  # (F, T)
        specs.append(torch.from_numpy(S))
    S = torch.stack(specs, dim=0)  # (B, F, T)
    return S.unsqueeze(1).contiguous()  # (B,1,F,T)

def run_one_epoch(model, loader, cfg: TrainCfg, train=True) -> Tuple[float, float]:
    model.train(mode=train)
    total_loss, total_corr, total_n = 0.0, 0, 0
    rng = np.random.default_rng(cfg.seed + (0 if train else 999))
    for wav, y, _ in loader:
        wav = wav.to(cfg.device)
        y = y.to(cfg.device).float()  # (B,)
        X = batch_to_spec(wav, cfg.sr, cfg.n_mels).to(cfg.device)  # (B,1,F,T)

        if cfg.use_specaug and train:
            X = specaug(X, rng)

        if cfg.mixup_alpha > 0 and train:
            X, y = mixup_batch(X, y, alpha=cfg.mixup_alpha, rng=rng)

        logits = model(X)  # expect (B,1) or (B,)
        if logits.ndim == 2 and logits.size(1) == 1: logits = logits.squeeze(1)
        if logits.ndim == 0: logits = logits.unsqueeze(0)

        loss = F.binary_cross_entropy_with_logits(logits, y)

        if train:
            model.zero_grad(set_to_none=True)
            loss.backward()
            for p in model.parameters():
                if p.grad is not None and torch.isnan(p.grad).any():
                    p.grad = torch.zeros_like(p.grad)
            opt = run_one_epoch.opt
            opt.step()

        with torch.no_grad():
            prob = torch.sigmoid(logits)
            pred = (prob >= 0.5).long()
            total_corr += (pred.cpu() == y.long().cpu()).sum().item()
            total_loss += float(loss.item()) * y.size(0)
            total_n += y.size(0)
    return total_loss / max(1, total_n), total_corr / max(1, total_n)

def evaluate_full(model, loader, cfg: TrainCfg) -> Dict[str, float]:
    model.eval()
    ys, ps = [], []
    total_loss, total_n = 0.0, 0
    for wav, y, _ in loader:
        wav = wav.to(cfg.device)
        y = y.to(cfg.device).float()
        X = batch_to_spec(wav, cfg.sr, cfg.n_mels).to(cfg.device)
        logits = model(X)
        if logits.ndim == 2 and logits.size(1) == 1: logits = logits.squeeze(1)
        if logits.ndim == 0: logits = logits.unsqueeze(0)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        prob = torch.sigmoid(logits)
        ys.append(y.detach().cpu().numpy())
        ps.append(prob.detach().cpu().numpy())
        total_loss += float(loss.item()) * y.size(0)
        total_n += y.size(0)
    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)
    m = compute_metrics(y_true, y_prob)
    m["loss"] = total_loss / max(1, total_n)
    m["y_true"] = y_true
    m["y_prob"] = y_prob
    return m

# attach optimizer to function scope (simple trick)
run_one_epoch.opt = None

# ---------- Plotting helpers ----------
def plot_mean_std(xs, Y, title, ylabel, out_png):
    # Y: list of (mean, std, label)
    plt.figure(figsize=(6,4))
    for (m, s, lbl) in Y:
        plt.plot(xs, m, label=lbl)
        plt.fill_between(xs, m - s, m + s, alpha=0.2)
    plt.xlabel("Epoch"); plt.ylabel(ylabel); plt.title(title); plt.legend()
    ensure_dir(Path(out_png).parent)
    plt.savefig(out_png); plt.close()
    print(f"Saved: {out_png}")

def plot_roc_pr(roc_list, pr_list, title_prefix, out_dir):
    plt.figure(figsize=(6,5))
    for auroc, fpr, tpr, lbl in roc_list:
        plt.plot(fpr, tpr, label=f"{lbl} (AUROC={auroc:.3f})")
    plt.plot([0,1], [0,1], "k--", linewidth=0.8)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"{title_prefix} – ROC")
    plt.legend(); ensure_dir(Path(out_dir)); plt.savefig(Path(out_dir)/"roc_curves.png"); plt.close()
    print(f"Saved: {Path(out_dir)/'roc_curves.png'}")

    plt.figure(figsize=(6,5))
    for aupr, prec, rec, lbl in pr_list:
        plt.plot(rec, prec, label=f"{lbl} (AUPR={aupr:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"{title_prefix} – PR")
    plt.legend(); plt.savefig(Path(out_dir)/"pr_curves.png"); plt.close()
    print(f"Saved: {Path(out_dir)/'pr_curves.png'}")

# ---------- Ablation driver ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with path,label[,patient_id]")
    ap.add_argument("--outdir", default="artifacts/ablation")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--model", default="aerocpdnet")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.2)
    args = ap.parse_args()

    set_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(args.csv)
    assert {"path","label"}.issubset(df.columns), "CSV must have path,label"

    out_root = Path(args.outdir); ensure_dir(out_root)

    # 1) Fixed Test split (stratified by label)
    tr_df, te_df = train_test_split(
        df, test_size=args.test_size, random_state=args.seed,
        stratify=df["label"].values
    )
    tr_df = tr_df.reset_index(drop=True)
    te_df = te_df.reset_index(drop=True)
    te_df.to_csv(out_root/"test_split.csv", index=False)

    # 2) Define ablations (change ONE factor each time)
    variants = [
        {"name": "Baseline (no aug)", "wave": False, "spec": False, "mix": 0.0},
        {"name": "+WaveAug",          "wave": True,  "spec": False, "mix": 0.0},
        {"name": "+SpecAug",          "wave": False, "spec": True,  "mix": 0.0},
        {"name": "+Mixup",            "wave": False, "spec": False, "mix": 0.2},
        {"name": "+Wave+Spec+Mix",    "wave": True,  "spec": True,  "mix": 0.2},
    ]

    # 3) K-Fold on TRAIN (not touching test)
    X = tr_df["path"].values
    y = tr_df["label"].values
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    # logs to aggregate
    global_summary = []

    for var in variants:
        vdir = out_root / var["name"].replace(" ", "_").replace("+","plus")
        ensure_dir(vdir)
        print(f"\n==== Variant: {var['name']} ====")

        # Store per-fold curves
        tr_acc_mat, va_acc_mat = [], []
        tr_loss_mat, va_loss_mat = [], []
        roc_list, pr_list = [], []

        fold_summaries = []
        for k, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
            print(f"Fold {k}/{args.folds}")
            fold_dir = vdir / f"fold{k}"; ensure_dir(fold_dir)

            tr_sub = tr_df.iloc[tr_idx].reset_index(drop=True)
            va_sub = tr_df.iloc[va_idx].reset_index(drop=True)

            ds_tr = CsvAudioDS(tr_sub, sr=args.sr, duration=args.duration,
                               use_waveaug=var["wave"], seed=args.seed+k)
            ds_va = CsvAudioDS(va_sub, sr=args.sr, duration=args.duration,
                               use_waveaug=False, seed=args.seed+k)
            ds_te = CsvAudioDS(te_df, sr=args.sr, duration=args.duration,
                               use_waveaug=False, seed=args.seed+k)

            dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=0)
            dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=0)
            dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=0)

            cfg = TrainCfg(
                model=args.model, sr=args.sr, duration=args.duration, n_mels=128,
                lr=args.lr, wd=args.wd, dropout=args.dropout, epochs=args.epochs,
                batch_size=args.batch_size, device=dev,
                use_waveaug=var["wave"], use_specaug=var["spec"],
                mixup_alpha=var["mix"], seed=args.seed+k
            )
            model = build_backbone(cfg.model, 1, 1, cfg.dropout).to(cfg.device)
            opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
            run_one_epoch.opt = opt

            rows = []
            best_score, best_state = -np.inf, None
            for epoch in range(1, cfg.epochs+1):
                tr_loss, tr_acc = run_one_epoch(model, dl_tr, cfg, train=True)
                va_metrics = evaluate_full(model, dl_va, cfg)
                rows.append({
                    "epoch": epoch,
                    "train_loss": tr_loss, "train_acc": tr_acc,
                    "val_loss": va_metrics["loss"], "val_acc": va_metrics["accuracy"],
                    "val_auroc": va_metrics["auroc"], "val_aupr": va_metrics["aupr"],
                    "val_f1": va_metrics["f1"], "val_precision": va_metrics["precision"],
                    "val_recall": va_metrics["recall"], "val_specificity": va_metrics["specificity"],
                    "val_bal_acc": va_metrics["balanced_accuracy"], "val_mcc": va_metrics["mcc"],
                    "val_brier": va_metrics["brier"]
                })
                score = va_metrics["auroc"] if not np.isnan(va_metrics["auroc"]) else va_metrics["f1"]
                if score > best_score:
                    best_score = score
                    best_state = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                                  for k, v in model.state_dict().items()}

                print(f"  ep{epoch:03d} | tr_loss {tr_loss:.4f} acc {tr_acc:.3f} | "
                      f"val_loss {va_metrics['loss']:.4f} acc {va_metrics['accuracy']:.3f} "
                      f"AUROC {va_metrics['auroc']:.3f} AUPR {va_metrics['aupr']:.3f}")

            df_ep = pd.DataFrame(rows)
            df_ep.to_csv(fold_dir/"epochs.csv", index=False)

            # curves collect
            tr_acc_mat.append(df_ep["train_acc"].values)
            va_acc_mat.append(df_ep["val_acc"].values)
            tr_loss_mat.append(df_ep["train_loss"].values)
            va_loss_mat.append(df_ep["val_loss"].values)

            # test using best
            if best_state is not None:
                model.load_state_dict(best_state)
            te_metrics = evaluate_full(model, dl_te, cfg)
            pd.Series(te_metrics).drop(labels=["fpr","tpr","prec","rec"], errors="ignore")\
                                 .to_csv(fold_dir/"test_metrics.csv")

            # curves for ROC/PR
            roc_list.append((te_metrics["auroc"], te_metrics["fpr"], te_metrics["tpr"], f"Fold {k}"))
            pr_list.append((te_metrics["aupr"], te_metrics["prec"], te_metrics["rec"], f"Fold {k}"))

            fold_summaries.append({
                "fold": k,
                "test_acc": te_metrics["accuracy"], "test_auroc": te_metrics["auroc"],
                "test_aupr": te_metrics["aupr"], "test_f1": te_metrics["f1"],
                "test_precision": te_metrics["precision"], "test_recall": te_metrics["recall"],
                "test_specificity": te_metrics["specificity"],
                "test_bal_acc": te_metrics["balanced_accuracy"], "test_mcc": te_metrics["mcc"],
                "test_brier": te_metrics["brier"], "test_loss": te_metrics["loss"],
            })

        # aggregate & plots for the variant
        df_fold = pd.DataFrame(fold_summaries)
        df_fold.to_csv(vdir/"test_summary_per_fold.csv", index=False)

        # mean±std epoch curves
        E = len(tr_acc_mat[0])
        xs = np.arange(1, E+1)
        def meanstd(mat):
            A = np.stack(mat, axis=0)
            return A.mean(axis=0), A.std(axis=0)
        m_tr_acc, s_tr_acc = meanstd(tr_acc_mat)
        m_va_acc, s_va_acc = meanstd(va_acc_mat)
        m_tr_ls, s_tr_ls = meanstd(tr_loss_mat)
        m_va_ls, s_va_ls = meanstd(va_loss_mat)

        plot_mean_std(xs,
            [(m_tr_acc, s_tr_acc, "Train"), (m_va_acc, s_va_acc, "Validation")],
            title=f"{var['name']} – Accuracy (mean±std over folds)",
            ylabel="Accuracy",
            out_png=vdir/"acc_curves_meanstd.png"
        )
        plot_mean_std(xs,
            [(m_tr_ls, s_tr_ls, "Train"), (m_va_ls, s_va_ls, "Validation")],
            title=f"{var['name']} – Loss (mean±std over folds)",
            ylabel="BCE Loss",
            out_png=vdir/"loss_curves_meanstd.png"
        )

        plot_roc_pr(roc_list, pr_list, title_prefix=var["name"], out_dir=vdir)

        # overall row for easy comparison
        row = {"variant": var["name"]}
        for k in ["test_acc","test_auroc","test_aupr","test_f1","test_precision","test_recall",
                  "test_specificity","test_bal_acc","test_mcc","test_brier","test_loss"]:
            row[f"{k}_mean"] = float(df_fold[k].mean())
            row[f"{k}_std"]  = float(df_fold[k].std())
        global_summary.append(row)

    pd.DataFrame(global_summary).to_csv(out_root/"_ablation_summary.csv", index=False)
    print(f"\nAblation summary: {out_root/'_ablation_summary.csv'}")
    print("Done.")

if __name__ == "__main__":
    main()
