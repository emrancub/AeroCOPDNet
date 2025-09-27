import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, matthews_corrcoef, brier_score_loss
)

def _ensure_4d(spec: torch.Tensor, batch_size: int) -> torch.Tensor:
    # Return (B,1,F,T), on whatever device spec is on.
    x = spec
    if x.ndim == 2:             # (F,T)
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 3:           # (B,F,T)
        x = x.unsqueeze(1)
    elif x.ndim == 4:           # (B,1,F,T) OK
        pass
    else:
        raise RuntimeError(f"Unexpected spec shape {spec.shape}")
    if x.size(0) != batch_size:
        # Batch dimension must match; if it's 1, likely (1,B,F,T) mistake upstream
        # Try to fix common mistake: treat first dim as channels when B>1
        if x.size(1) == batch_size:   # (1,B,F,T) -> transpose to (B,1,F,T)
            x = x.transpose(0,1).contiguous()
        else:
            raise RuntimeError(f"Batch mismatch after feature: got {x.shape}, expected B={batch_size}")
    return x

def _ensure_logit_shape(logits: torch.Tensor) -> torch.Tensor:
    # make logits (B,1)
    if logits.ndim == 1:
        logits = logits.unsqueeze(1)
    elif logits.ndim == 2 and logits.size(1) != 1:
        logits = logits[:, :1]
    elif logits.ndim > 2:
        logits = logits.view(logits.size(0), -1)[:, :1]
    return logits

def _bce_logits(logits, target, pos_weight=None):
    if pos_weight is not None:
        pw = torch.as_tensor([pos_weight], device=logits.device, dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)
    return F.binary_cross_entropy_with_logits(logits, target)

def _batch_from_loader(batch):
    # Support (wav, y) or (wav, y, path)
    if isinstance(batch, (list, tuple)) and len(batch) == 3:
        X_wav, y, _ = batch
    elif isinstance(batch, (list, tuple)) and len(batch) == 2:
        X_wav, y = batch
    else:
        raise RuntimeError("Expected batch to be (wav, y) or (wav, y, path)")
    return X_wav, y

def _mixup(spec: torch.Tensor, y: torch.Tensor, alpha: float):
    # spec: (B,1,F,T) on CPU, y: (B,) or (B,1)
    if not alpha or alpha <= 0:
        return spec, y.float().view(-1, 1)

    B = spec.size(0)
    # np.random.beta returns a scalar; cast safely to Python float
    lam = float(np.random.beta(alpha, alpha))

    idx = torch.randperm(B)              # CPU
    y = y.float().view(-1, 1)            # (B,1)

    # use a torch scalar for broadcasting with tensors
    lam_t = torch.tensor(lam, dtype=spec.dtype)

    spec_m = lam_t * spec + (1.0 - lam_t) * spec[idx]
    y_m = lam * y + (1.0 - lam) * y[idx]
    return spec_m, y_m


def train_one_epoch(model, loader, optimizer, feature, specaug, device, pos_weight=None, mixup_alpha=0.0):
    model.train()
    total_loss = 0.0
    total_correct = 0
    n_samples = 0

    for batch in loader:
        X_wav, y = _batch_from_loader(batch)     # all on CPU here
        B = X_wav.size(0)

        # 1) features on CPU
        X_spec = feature(X_wav)                  # -> (B,1,F,T) CPU
        X_spec = _ensure_4d(X_spec, B)

        # 2) mixup on CPU (before masking)
        X_spec, y_t = _mixup(X_spec, y, mixup_alpha)

        # 3) spec augment on CPU
        if specaug is not None:
            X_spec = specaug(X_spec)             # still (B,1,F,T) CPU

        # 4) move to device
        X_spec = X_spec.to(device, non_blocking=True).float()
        y_t = y_t.to(device).float()             # (B,1)

        # forward + loss
        logits = _ensure_logit_shape(model(X_spec))  # (B,1)
        loss = _bce_logits(logits, y_t, pos_weight)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # metrics
        with torch.no_grad():
            prob = torch.sigmoid(logits).squeeze(1)       # (B,)
            y_hat = (prob >= 0.5).long()
            y_int = y_t.squeeze(1).long()
            total_correct += (y_hat == y_int).sum().item()
            total_loss += loss.item() * B
            n_samples += B

    tr_loss = total_loss / max(1, n_samples)
    tr_acc = total_correct / max(1, n_samples)
    return tr_loss, tr_acc

@torch.no_grad()
def evaluate(model, loader, feature, device="cpu"):
    model.eval()

    all_logits = []
    all_y = []
    total_loss = 0.0
    n = 0

    for batch in loader:
        X_wav, y = _batch_from_loader(batch)
        B = X_wav.size(0)

        X_spec = feature(X_wav)                  # CPU
        X_spec = _ensure_4d(X_spec, B)

        X_spec = X_spec.to(device, non_blocking=True).float()
        y = torch.as_tensor(y, device=device).float().view(-1,1)

        logits = _ensure_logit_shape(model(X_spec))  # (B,1)
        loss = F.binary_cross_entropy_with_logits(logits, y)

        total_loss += loss.item() * logits.size(0)
        n += logits.size(0)
        all_logits.append(logits.detach().cpu())
        all_y.append(y.detach().cpu())

    if n == 0:
        return {
            "loss": float("nan"), "accuracy": float("nan"),
            "precision": float("nan"), "recall": float("nan"),
            "f1": float("nan"), "sensitivity": float("nan"),
            "specificity": float("nan"), "balanced_accuracy": float("nan"),
            "auc_roc": float("nan"), "aupr": float("nan"),
            "mcc": float("nan"), "brier": float("nan"),
            "y_true": [], "y_prob": [], "fpr": [], "tpr": [],
            "prec": [], "rec": []
        }

    logits = torch.cat(all_logits, dim=0).squeeze(1).numpy()  # (N,)
    y_true = torch.cat(all_y, dim=0).squeeze(1).numpy().astype(int)  # (N,)
    y_prob = 1.0 / (1.0 + np.exp(-logits))
    y_hat = (y_prob >= 0.5).astype(int)

    # metrics
    acc = accuracy_score(y_true, y_hat)
    f1 = f1_score(y_true, y_hat, zero_division=0)
    prec = precision_score(y_true, y_hat, zero_division=0)
    rec = recall_score(y_true, y_hat, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bal_acc = 0.5 * (sens + spec) if ((tp + fn) > 0 and (tn + fp) > 0) else acc

    try:
        auroc = roc_auc_score(y_true, y_prob)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
    except Exception:
        auroc = float("nan"); fpr = np.array([]); tpr = np.array([])

    try:
        aupr = average_precision_score(y_true, y_prob)
        prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_prob)
    except Exception:
        aupr = float("nan"); prec_curve = np.array([]); rec_curve = np.array([])

    try:
        mcc = matthews_corrcoef(y_true, y_hat)
    except Exception:
        mcc = float("nan")

    try:
        brier = brier_score_loss(y_true, y_prob)
    except Exception:
        brier = float("nan")

    return {
        "loss": total_loss / n,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "sensitivity": sens,
        "specificity": spec,
        "balanced_accuracy": bal_acc,
        "auc_roc": auroc,
        "aupr": aupr,
        "mcc": mcc,
        "brier": brier,
        "y_true": y_true.tolist(),
        "y_prob": y_prob.tolist(),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "prec": prec_curve.tolist(),
        "rec": rec_curve.tolist(),
        # aliases
        "auroc": auroc,
        "average_precision": aupr,
    }
