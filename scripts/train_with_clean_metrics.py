# scripts/train_with_clean_metrics.py
import math, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# ---- optional EMA (no extra dependency) ----
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {k: p.clone().detach() for k, p in model.state_dict().items()}
        self.backup = None
    @torch.no_grad()
    def update(self):
        for k, p in self.model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(p, alpha=1.0 - self.decay)
    def store(self):
        self.backup = {k: p.clone() for k, p in self.model.state_dict().items()}
    @torch.no_grad()
    def copy_to(self):
        for k, p in self.model.state_dict().items():
            p.copy_(self.shadow[k])
    def restore(self):
        if self.backup is not None:
            self.model.load_state_dict(self.backup); self.backup = None

# ---- metric helpers ----
def bce_loss(logits, targets, weight_pos=None, weight_neg=None):
    if weight_pos is None or weight_neg is None:
        return F.binary_cross_entropy_with_logits(logits, targets)
    w = torch.where(targets > 0.5, torch.as_tensor(weight_pos, device=logits.device),
                                     torch.as_tensor(weight_neg, device=logits.device))
    return F.binary_cross_entropy_with_logits(logits, targets, weight=w)

@torch.no_grad()
def evaluate_clean(model, loader, feature, device, use_ema: EMA = None, thresh=0.5):
    """Evaluate on CLEAN (no-augment) data to make training vs validation comparable."""
    if use_ema is not None:
        use_ema.store(); use_ema.copy_to()
    model.eval()
    n, loss_sum = 0, 0.0
    tp = tn = fp = fn = 0
    for wav, y in loader:                       # expects (wave, label)
        y = y.float().to(device).view(-1, 1)
        X = feature(wav).to(device).float()     # (B,1,F,T)
        logits = model(X)
        loss_sum += F.binary_cross_entropy_with_logits(logits, y).item() * y.size(0)
        p = torch.sigmoid(logits)
        pred = (p >= thresh).float()
        tp += ((pred == 1) & (y == 1)).sum().item()
        tn += ((pred == 0) & (y == 0)).sum().item()
        fp += ((pred == 1) & (y == 0)).sum().item()
        fn += ((pred == 0) & (y == 1)).sum().item()
        n += y.size(0)
    if use_ema is not None:
        use_ema.restore()
    acc = (tp + tn) / max(1, n)
    return loss_sum / max(1, n), acc

# ---- core training loop ----
def fit(model, feature, train_loader, train_eval_loader, val_loader,
        optimizer, scheduler=None, ema_decay=0.999, epochs=100, device="cuda",
        mixup_alpha=0.2, class_weights=None, out_csv="epochs.csv"):

    ema = EMA(model, ema_decay) if ema_decay else None
    history = []

    def do_mixup(x, y, alpha):
        if alpha <= 0: return x, y
        lam = np.random.beta(alpha, alpha)
        idx = torch.randperm(x.size(0), device=x.device)
        x_mix = lam * x + (1 - lam) * x[idx]
        y_mix = lam * y + (1 - lam) * y[idx]
        return x_mix, y_mix

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0; nseen = 0
        for wav, y in train_loader:
            y = y.float().view(-1, 1).to(device)        # hard labels 0/1
            # ---- STRONG AUGMENTS on waveform happen in the dataset/dataloader ----
            X = feature(wav).to(device).float()          # (B,1,F,T)
            # Mixup in feature space (or do it in waveform before feature())
            if mixup_alpha and mixup_alpha > 0:
                X, y_soft = do_mixup(X, y, mixup_alpha)
            else:
                y_soft = y

            logits = model(X)
            loss = bce_loss(logits, y_soft, *(class_weights or (None, None)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if ema: ema.update()

            bs = y.size(0)
            running += loss.item() * bs
            nseen += bs

        # ---- epoch-end CLEAN evaluations (no augment) ----
        tr_loss_clean, tr_acc_clean = evaluate_clean(model, train_eval_loader, feature, device, use_ema=ema)
        val_loss, val_acc = evaluate_clean(model, val_loader, feature, device, use_ema=ema)

        if scheduler is not None:
            # ReduceLROnPlateau should step on validation loss, others step every epoch
            if hasattr(scheduler, "step") and "plateau" in scheduler.__class__.__name__.lower():
                scheduler.step(val_loss)
            else:
                scheduler.step()

        row = dict(epoch=epoch,
                   train_loss_batch=running / max(1, nseen),   # loss on augmented batches
                   train_loss=tr_loss_clean,                   # clean-train loss
                   train_acc=tr_acc_clean,                     # <-- comparable to val_acc
                   val_loss=val_loss,
                   val_acc=val_acc)
        history.append(row)
        print(f"[{epoch:03d}] clean-train acc={tr_acc_clean:.3f} val acc={val_acc:.3f} "
              f"| clean-train loss={tr_loss_clean:.3f} val loss={val_loss:.3f}")

    df = pd.DataFrame(history)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df
