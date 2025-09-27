# scripts/cv_train.py
import argparse, os, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.model_selection import StratifiedKFold
from src.copd.utils import ensure_dir, set_seed, device_str, timestamp
from src.copd.data import AudioBinaryDataset, build_sampler
from src.copd.features import build_feature
from src.copd.augment import SpecAugment
# from src.copd.old_models import build_model
from src.copd.models import build_model
from src.copd.trainloop import train_one_epoch, evaluate

def run_fold(fold, tr_idx, va_idx, ds, args, dev, run_dir):
    tr_ds = torch.utils.data.Subset(ds, tr_idx)
    va_ds = torch.utils.data.Subset(ds, va_idx)

    y_tr = np.array([ds[i][1] for i in tr_idx], int)
    sampler = None if args.no_balance else build_sampler(y_tr)

    tr_loader = torch.utils.data.DataLoader(
        tr_ds, batch_size=args.batch_size,
        shuffle=(sampler is None), sampler=sampler, num_workers=0)

    va_loader = torch.utils.data.DataLoader(
        va_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    feat = build_feature(args.features)
    specaug = SpecAugment() if args.use_specaug else None

    model = build_model(args.model, in_ch=1, n_classes=1, dropout=args.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = -1; rows=[]
    ckpt = run_dir / f"fold{fold}_best.pt"
    csv  = run_dir / f"epochs_fold{fold}.csv"

    for ep in range(1, args.epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, tr_loader, opt, feat, specaug, dev,
                                          pos_weight=None, mixup_alpha=args.mixup)
        val = evaluate(model, va_loader, feat, device=dev)
        rows.append({
            "epoch":ep, "train_loss":tr_loss, "train_acc":tr_acc,
            "val_loss":val["loss"], "val_acc":val["accuracy"], "val_auc":val["auc_roc"],
            "val_aupr":val["aupr"], "val_f1":val["f1"],
            "val_sens":val["sensitivity"], "val_spec":val["specificity"],
            "val_bal_acc":val["balanced_accuracy"], "val_prec":val["precision"],
            "val_recall":val["recall"], "val_mcc":val["mcc"], "val_brier":val["brier"],
        })
        score = val["auc_roc"] if not np.isnan(val["auc_roc"]) else val["f1"]
        if score > best:
            best = score
            torch.save(model.state_dict(), ckpt)

            # save preds for this best state (for ROC/PR/ConfMat plots)
            preds = pd.DataFrame({
                "path":[ds[i][2] if len(ds[i])==3 else "" for i in va_idx],
                "label":val["y_true"], "prob":val["y_prob"]
            })
            preds.to_csv(run_dir / f"fold{fold}_preds.csv", index=False)

            # save ROC/PR points too
            np.savez(run_dir / f"fold{fold}_curves.npz",
                     fpr=val["fpr"], tpr=val["tpr"], prec=val["prec"], rec=val["rec"])

    pd.DataFrame(rows).to_csv(csv, index=False)
    return rows[-1]  # last epoch row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--mixup", type=float, default=0.2)
    ap.add_argument("--use_specaug", action="store_true")
    ap.add_argument("--no_balance", action="store_true")
    ap.add_argument("--features", default="mel", choices=["mel","mfcc"])
    # ap.add_argument("--model", default="aerocpdnet",
    #                 choices=["basiccnn","crnn","lstm","bilstm","gru","aerocpdnet"])
    ap.add_argument(
        "--model",
        default="aerocpdnet",
        choices=["aerocpdnet", "aerocpdnetlite", "basiccnn", "crnn", "lstm", "gru", "resnet18"],
    )
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed); dev = device_str()

    ds = AudioBinaryDataset(args.csv, sample_rate=args.sr, duration=args.duration, return_path=False)
    y = np.array([ds[i][1] for i in range(len(ds))], int)

    # Stratified split (patient_id column is optional in your CSV)
    kf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    run_dir = Path("artifacts/cv_old") / f"cv_{timestamp()}_{args.folds}f"
    ensure_dir(run_dir)

    last_rows=[]; fold=1
    for tr_idx, va_idx in kf.split(np.zeros_like(y), y):
        print(f"Fold {fold}/{args.folds}: train={len(tr_idx)}, val={len(va_idx)}")
        last_rows.append(run_fold(fold, tr_idx, va_idx, ds, args, dev, run_dir))
        fold += 1

    # Summary
    df = pd.DataFrame(last_rows)
    summ = df.mean(numeric_only=True).to_dict()
    pd.DataFrame([summ]).to_csv(run_dir / "summary.csv", index=False)
    print("Summary:", summ)
    print(f"All artifacts in: {run_dir}")

if __name__ == "__main__":
    main()
