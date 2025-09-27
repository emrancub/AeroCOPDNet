import argparse, os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
import pandas as pd
from src.copd.utils import ensure_dir, set_seed, device_str, timestamp
from src.copd.data import AudioBinaryDataset, build_sampler
from src.copd.augment import WaveAugment, SpecAugment
from src.copd.features import build_feature
from src.copd.old_models import build_model
from src.copd.trainloop import train_one_epoch, evaluate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Labels CSV with path,label[,patient_id]")
    ap.add_argument("--model", default="aerocpdnet", choices=["basiccnn","crnn","lstm","bilstm","gru","aerocpdnet"])
    ap.add_argument("--features", default="mel", choices=["mel","mfcc"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--mixup", type=float, default=0.0)
    ap.add_argument("--use_specaug", action="store_true")
    ap.add_argument("--no_class_weights", action="store_true")
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--duration", type=float, default=4.0)
    args = ap.parse_args()

    set_seed(args.seed)
    dev = device_str()
    print(f"Device: {dev}")

    ds = AudioBinaryDataset(args.csv, sample_rate=args.sr, duration=args.duration, return_path=True)
    N = len(ds)
    n_val = max(1, int(args.val_split * N))
    n_train = N - n_val
    tr_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    # class weights & samplers
    tr_labels = np.array([ds[ i ][1] for i in tr_ds.indices], dtype=int)
    if args.no_class_weights:
        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
        pos_weight = None
    else:
        sampler = build_sampler(tr_labels)
        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0, drop_last=False)
        # pos_weight for BCE
        c = np.bincount(tr_labels, minlength=2); pos_weight = c[0] / max(1, c[1])

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    # features & augment
    feat = build_feature(args.features)      # returns CPU tensors; trainloop moves to CUDA
    specaug = SpecAugment() if args.use_specaug else None

    # model
    model = build_model(args.model, in_ch=1, n_classes=1, dropout=args.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # logging
    run_id = f"{args.model}_{timestamp()}"
    out_csv = Path(f"artifacts/logs/epochs_{run_id}.csv"); ensure_dir(out_csv.parent)
    out_ckpt = Path(f"artifacts/models/{run_id}_best.pt"); ensure_dir(out_ckpt.parent)

    best_auroc = -1
    rows=[]
    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, tr_loader, opt, feat, specaug, dev, pos_weight, mixup_alpha=args.mixup)
        val_stats = evaluate(model, val_loader, feat, device=dev)
        # --- robust metric getters so we don't crash on missing keys ---
        acc_val = float(val_stats.get("accuracy", val_stats.get("acc", float("nan"))))
        auc_val = float(val_stats.get("auc_roc", val_stats.get("auc", float("nan"))))
        aupr_val = float(val_stats.get("aupr", val_stats.get("average_precision", float("nan"))))
        f1_val = float(val_stats.get("f1", float("nan")))
        loss_val = float(val_stats.get("loss", float("nan")))

        # be robust to key naming
        val_auc = (
            val_stats.get("auc_roc", None)
            if isinstance(val_stats, dict) else None
        )
        if val_auc is None:
            val_auc = val_stats.get("auroc", None) if isinstance(val_stats, dict) else None
        if val_auc is None:
            val_auc = float("nan")

        row = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": loss_val,
            "val_acc": acc_val,
            "val_auc": auc_val,
            "val_aupr": aupr_val,
            "val_f1": f1_val,
            "val_sens": float(val_stats.get("sensitivity", float("nan"))),
            "val_spec": float(val_stats.get("specificity", float("nan"))),
            "val_bal_acc": float(val_stats.get("balanced_accuracy", float("nan"))),
        }

        rows.append(row)
        acc_val = val_stats.get("accuracy", val_stats.get("acc", float("nan")))
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={tr_loss:.4f} "
            f"val_loss={loss_val:.4f} "
            f"AUC={auc_val:.4f} "
            f"AUPR={aupr_val:.4f} "
            f"F1={f1_val:.3f} "
            f"acc={acc_val:.3f}"
        )

        # save best by AUC if available else F1
        score = val_stats["auc_roc"];
        if score != score:  # NaN
            score = val_stats["f1"]
        if score > best_auroc:
            best_auroc = score
            torch.save(model.state_dict(), out_ckpt)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Epoch CSV: {out_csv}")
    print(f"Best model: {out_ckpt}")

if __name__=="__main__":
    main()
