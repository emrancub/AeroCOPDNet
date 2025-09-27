import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from src.copd.utils import ensure_dir, set_seed, device_str, timestamp
from src.copd.data import AudioBinaryDataset, build_sampler
from src.copd.augment import SpecAugment
from src.copd.features import FeatureExtractor
from src.copd.old_models import build_model
from src.copd.trainloop import train_one_epoch, evaluate
from src.copd.cv import build_folds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--model", default="aerocpdnet", choices=["basiccnn","crnn","lstm","bilstm","gru","aerocpdnet"])
    ap.add_argument("--features", default="mel", choices=["mel","mfcc"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--mixup", type=float, default=0.0)
    ap.add_argument("--use_specaug", action="store_true")
    ap.add_argument("--no_class_weights", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--duration", type=float, default=4.0)
    args = ap.parse_args()

    set_seed(args.seed)
    dev = device_str()
    print(f"Device: {dev}")

    df, fold_idx = build_folds(args.csv, n_splits=args.folds, seed=args.seed)
    feat = FeatureExtractor(kind=args.features, sr=args.sr)
    specaug = SpecAugment() if args.use_specaug else None

    cv_rows=[]
    for k,(tr_ids,val_ids) in enumerate(fold_idx, start=1):
        print(f"[Fold {k}/{args.folds}] train={len(tr_ids)} val={len(val_ids)}")
        ds = AudioBinaryDataset(args.csv, sample_rate=args.sr, duration=args.duration, return_path=True)
        tr_ds = Subset(ds, tr_ids); val_ds = Subset(ds, val_ids)
        tr_labels = np.array([ds[i][1] for i in tr_ids], dtype=int)

        if args.no_class_weights:
            tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
            pos_weight=None
        else:
            sampler = build_sampler(tr_labels)
            tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0)
            c = np.bincount(tr_labels, minlength=2); pos_weight = c[0] / max(1, c[1])
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

        model = build_model(args.model, in_ch=1, n_classes=1, dropout=args.dropout).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_auc=-1; fold_rows=[]
        for epoch in range(1, args.epochs+1):
            tr_loss, tr_acc = train_one_epoch(model, tr_loader, opt, feat, specaug, dev, pos_weight, mixup_alpha=args.mixup)
            val_stats = evaluate(model, val_loader, feat, specaug=None, device=dev)
            fold_rows.append({"fold":k,"epoch":epoch,"train_loss":tr_loss,"train_acc":float(tr_acc),
                              "val_loss":val_stats["loss"],"val_auc":val_stats["auc_roc"],"val_aupr":val_stats["aupr"],
                              "val_acc":val_stats["accuracy"],"val_f1":val_stats["f1"],
                              "val_sens":val_stats["sensitivity"],"val_spec":val_stats["specificity"]})
            if val_stats["auc_roc"]==val_stats["auc_roc"] and val_stats["auc_roc"]>best_auc:
                best_auc=val_stats["auc_roc"]
                ckpt = Path(f"artifacts/models/{args.model}_fold{k}_{timestamp()}_best.pt")
                ensure_dir(ckpt.parent); torch.save(model.state_dict(), ckpt)
        pd.DataFrame(fold_rows).to_csv(f"artifacts/logs/cv_fold{k}_{args.model}.csv", index=False)
        last = fold_rows[-1]
        cv_rows.append({"fold":k, "best_auc":best_auc, "last_val_acc":last["val_acc"], "last_val_f1":last["val_f1"]})
    pd.DataFrame(cv_rows).to_csv(f"artifacts/reports/cv_summary_{args.model}.csv", index=False)
    print("CV summary saved to artifacts/reports")

if __name__=="__main__":
    main()
