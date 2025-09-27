# scripts/eval_csv.py
import argparse, os, json
import torch, torchaudio
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             accuracy_score, balanced_accuracy_score,
                             confusion_matrix, roc_curve, precision_recall_curve)

# --- try to reuse your project feature/model builders
try:
    from src.copd.features import FeatureExtractor  # our project class
except Exception:
    FeatureExtractor = None

from src.copd.old_models import build_model  # used by Runing.train.py


# ----------------- minimal dataset + collate (self-contained) -----------------
class BinaryCsvDataset(Dataset):
    """Reads a CSV with columns [path|filepath], [label].
       Returns (waveform[1,T], label_float)."""
    def __init__(self, csv_path, target_sr=16000):
        df = pd.read_csv(csv_path)
        path_col = "path" if "path" in df.columns else "filepath"
        if path_col not in df.columns or "label" not in df.columns:
            raise RuntimeError(f"CSV must have columns 'path' (or 'filepath') and 'label'. Got {df.columns.tolist()}")
        self.paths = df[path_col].astype(str).tolist()
        self.labels = df["label"].astype(float).tolist()
        self.target_sr = target_sr

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        wav, sr = torchaudio.load(self.paths[i])  # (C,T)
        if wav.shape[0] > 1:  # mono
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)
        return wav, float(self.labels[i])

def pad_collate(batch):
    """Pads variable-length spectrograms in time dimension."""
    Xs, ys = zip(*batch)  # X is (1,F,T) **after feature step**; see below
    # if still raw waveforms, we’ll feature-ize later (handled in loop)
    return list(Xs), torch.tensor(ys).float()


# ------------------------- feature extractor fallback -------------------------
class _MelIfNeeded:
    """If your project FeatureExtractor is available we’ll use it.
       Otherwise use this clean mel-spectrogram -> log-mel -> z-score."""
    def __init__(self, name="mel", sr=16000, n_mels=128, n_fft=1024, hop=256):
        self.name = name
        if FeatureExtractor is not None:
            self.fx = FeatureExtractor(name=name, sample_rate=sr)
        else:
            self.fx = None
            self.mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels
            )
            self.db  = torchaudio.transforms.AmplitudeToDB()

    def __call__(self, wav_1_T):
        if self.fx is not None:
            S = self.fx(wav_1_T)              # expected (1,F,T)
        else:
            S = self.mel(wav_1_T)             # (1,F,T)
            S = self.db(S)
            S = (S - S.mean()) / (S.std() + 1e-6)
        # ensure shape (1,F,T)
        if S.dim() == 2: S = S.unsqueeze(0)
        return S


# -------------------------------- evaluation ---------------------------------
@torch.no_grad()
def evaluate_csv(csv_path, ckpt, model_name, features="mel", batch_size=16, device="cuda", out_prefix=None):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    ds = BinaryCsvDataset(csv_path)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=pad_collate, num_workers=0)

    # build model & load weights (same as in Runing.train.py)
    model = build_model(model_name, in_ch=1, n_classes=1, dropout=0.0).to(device)
    state = torch.load(ckpt, map_location=device)
    if "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()

    feat = _MelIfNeeded(name=features)
    all_logits, all_probs, all_labels = [], [], []

    for wav_list, y in loader:
        # make a batch of features with padding in time
        feats = [feat(w).to(device) for w in wav_list]         # list of (1,F,T)
        F = feats[0].shape[1]
        T = max(f.shape[-1] for f in feats)
        X = torch.zeros(len(feats), 1, F, T, device=device)
        for i, f in enumerate(feats):
            X[i, :, :, :f.shape[-1]] = f

        y = y.to(device)
        logits = model(X).view(-1)                  # (B,)
        probs  = torch.sigmoid(logits)
        all_logits.append(logits.detach().cpu())
        all_probs.append(probs.detach().cpu())
        all_labels.append(y.detach().cpu())

    y_true = torch.cat(all_labels).numpy().astype(int)
    y_prob = torch.cat(all_probs).numpy()
    y_pred = (y_prob >= 0.5).astype(int)

    # metrics
    acc   = accuracy_score(y_true, y_pred)
    bacc  = balanced_accuracy_score(y_true, y_pred)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc  = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    try:
        aupr = average_precision_score(y_true, y_prob)
    except ValueError:
        aupr = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    sens = tp / (tp + fn + 1e-9)  # recall for class 1
    spec = tn / (tn + fp + 1e-9)

    # figures
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,4))

    # ROC
    plt.subplot(1,3,1)
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.plot(fpr, tpr)
    except ValueError:
        plt.plot([0,1],[0,1], linestyle="--")
    plt.title(f"ROC (AUC={auc:.3f})"); plt.xlabel("FPR"); plt.ylabel("TPR")

    # PR
    plt.subplot(1,3,2)
    try:
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        plt.plot(rec, prec)
    except ValueError:
        plt.plot([0,1],[np.mean(y_true)]*2, linestyle="--")
    plt.title(f"PR (AUPR={aupr:.3f})"); plt.xlabel("Recall"); plt.ylabel("Precision")

    # Confusion matrix
    plt.subplot(1,3,3)
    cm = np.array([[tn, fp],[fn, tp]])
    im = plt.imshow(cm, cmap="Blues")
    for (i,j),v in np.ndenumerate(cm):
        plt.text(j,i,str(int(v)), ha="center", va="center")
    plt.xticks([0,1], ["Non-COPD","COPD"]); plt.yticks([0,1], ["Non-COPD","COPD"])
    plt.title(f"Confusion\nacc={acc:.3f} bacc={bacc:.3f}\nsens={sens:.3f} spec={spec:.3f}")
    plt.tight_layout()

    # outputs
    if out_prefix is None:
        base = os.path.splitext(os.path.basename(csv_path))[0]
        out_prefix = os.path.join("artifacts", "reports", f"test_{base}")
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

    fig_path = out_prefix + "_roc_pr_cm.png"
    plt.savefig(fig_path, dpi=200)

    metrics = dict(
        accuracy=acc, balanced_accuracy=bacc, f1=f1, auc_roc=auc, aupr=aupr,
        sensitivity=sens, specificity=spec, tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)
    )
    with open(out_prefix + "_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    pred_csv = out_prefix + "_preds.csv"
    pd.DataFrame({"path": ds.paths, "label": y_true, "prob": y_prob, "pred": y_pred}).to_csv(pred_csv, index=False)

    print("\n=== TEST RESULTS ===")
    for k,v in metrics.items():
        if isinstance(v, float): print(f"{k:>18s}: {v:.4f}")
        else:                    print(f"{k:>18s}: {v}")
    print(f"\nSaved: {fig_path}\nSaved: {out_prefix + '_metrics.json'}\nSaved: {pred_csv}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Test CSV with columns path,label")
    ap.add_argument("--ckpt", required=True, help="Path to .pt checkpoint")
    ap.add_argument("--model", required=True, choices=["aerocpdnet","cnnsmall","resnet18","lstm"], help="Model name (as in old_models.py)")
    ap.add_argument("--features", default="mel", help="Feature type (default: mel)")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_prefix", default=None)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    evaluate_csv(args.csv, args.ckpt, args.model, args.features, args.batch_size, args.device, args.out_prefix)
