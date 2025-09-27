# scripts/posthoc_trainval_from_best.py
import argparse, json, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold

# your project imports
from src.copd.utils import ensure_dir, device_str, set_seed
from src.copd.data import AudioBinaryDataset
from src.copd.features import build_feature
from src.copd.models import build_model
from src.copd.trainloop import evaluate

# ---------- helpers ----------
def metrics_get(d, key, default=np.nan):
    return float(d.get(key, d.get(key.replace("auc_roc","auc"), default)))

def stratified_folds(df, k=5, seed=42):
    y = df["label"].astype(int).values
    idx = np.arange(len(df))
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for tr_idx, va_idx in skf.split(idx, y):
        yield tr_idx, va_idx

def load_best(run_dir: Path, fold: int, device="cpu"):
    ckpt = run_dir / f"fold_{fold:02d}" / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing: {ckpt}")
    state = torch.load(ckpt, map_location=device)
    return state, ckpt

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True,
                    help=r'Folder like artifacts\cv\cvtest_YYYYMMDD_HHMMSS_5f')
    ap.add_argument("--model", required=True, help="model name used (e.g., resnet18, aerocpdnetlite, ...)")
    ap.add_argument("--features", default="mel")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--select_by", choices=["val_acc","val_loss"], default="val_acc",
                    help="criterion to choose the single best fold")
    ap.add_argument("--copy_best_as", default="best_overall.pt",
                    help="filename to copy the best fold checkpoint to (in run_dir)")
    args = ap.parse_args()

    set_seed(args.seed)
    dev = device_str()
    run_dir = Path(args.run_dir)
    ensure_dir(run_dir)

    # the pool CSV saved by your cv_train_with_test run
    pool_csv = run_dir / "trainval_pool.csv"
    if not pool_csv.exists():
        raise SystemExit(f"Not found: {pool_csv}. Run cv_train_with_test first.")

    df_pool = pd.read_csv(pool_csv)
    base_ds = AudioBinaryDataset(str(pool_csv), sample_rate=args.sr, duration=args.duration, return_path=False)

    feature = build_feature(args.features)  # no augment here
    B = args.batch_size

    rows = []
    for fold, (tr_idx, va_idx) in enumerate(stratified_folds(df_pool, k=args.folds, seed=args.seed), start=1):
        fold_dir = run_dir / f"fold_{fold:02d}"
        ensure_dir(fold_dir)

        # Build the SAME architecture used in training
        model = build_model(args.model, in_ch=1, n_classes=1, dropout=0.0).to(dev)
        state, ckpt_path = load_best(run_dir, fold, device="cpu")
        # forgiving load: keep whatever matches (works for all your models here)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[fold{fold:02d}] load: missing={len(missing)} unexpected={len(unexpected)} (ok)")

        model.eval()

        tr_loader = DataLoader(Subset(base_ds, tr_idx.tolist()), batch_size=B, shuffle=False, num_workers=0)
        va_loader = DataLoader(Subset(base_ds, va_idx.tolist()), batch_size=B, shuffle=False, num_workers=0)

        # IMPORTANT: evaluate **training split with no augmentation**
        tr_stats = evaluate(model, tr_loader, feature, device=dev)
        va_stats = evaluate(model, va_loader, feature, device=dev)

        out = dict(
            fold=fold,
            train_loss=metrics_get(tr_stats, "loss"),
            train_acc =metrics_get(tr_stats, "accuracy", tr_stats.get("acc", np.nan)),
            val_loss  =metrics_get(va_stats, "loss"),
            val_acc   =metrics_get(va_stats, "accuracy", va_stats.get("acc", np.nan)),
            ckpt=str(ckpt_path)
        )
        rows.append(out)

        # save per-fold json
        with open(fold_dir / "posthoc_trainval.json", "w") as f:
            json.dump(out, f, indent=2)

        print(f"[fold {fold}] train_acc={out['train_acc']:.4f}  val_acc={out['val_acc']:.4f}  "
              f"train_loss={out['train_loss']:.4f}  val_loss={out['val_loss']:.4f}")

    # Table + mean±std
    df = pd.DataFrame(rows).sort_values("fold")
    df.to_csv(run_dir / "posthoc_train_val_summary.csv", index=False)

    def ms(col):
        v = df[col].values.astype(float)
        return float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0

    m_tr_acc, s_tr_acc = ms("train_acc")
    m_va_acc, s_va_acc = ms("val_acc")
    m_tr_lo , s_tr_lo  = ms("train_loss")
    m_va_lo , s_va_lo  = ms("val_loss")

    print("\n===== PER-FOLD (train vs val) SUMMARY =====")
    print(df[["fold","train_acc","val_acc","train_loss","val_loss"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n===== MEAN ± STD ACROSS FOLDS =====")
    print(f"train_acc : {m_tr_acc:.4f} ± {s_tr_acc:.4f}")
    print(f"val_acc   : {m_va_acc:.4f} ± {s_va_acc:.4f}")
    print(f"train_loss: {m_tr_lo :.4f} ± {s_tr_lo :.4f}")
    print(f"val_loss  : {m_va_lo :.4f} ± {s_va_lo :.4f}")

    # Select best fold
    if args.select_by == "val_acc":
        best_row = df.iloc[df["val_acc"].values.argmax()]
    else:
        best_row = df.iloc[df["val_loss"].values.argmin()]

    best_fold = int(best_row["fold"])
    best_ckpt = run_dir / f"fold_{best_fold:02d}" / "best.pt"
    copy_to  = run_dir / args.copy_best_as
    shutil.copy2(best_ckpt, copy_to)

    with open(run_dir / "best_fold.txt", "w") as f:
        f.write(f"best_fold={best_fold}\ncriterion={args.select_by}\nckpt={best_ckpt}\n")

    print(f"\n>>> BEST FOLD: {best_fold}  (criterion: {args.select_by})")
    print(f"    Acc/Loss: val_acc={best_row['val_acc']:.4f}  val_loss={best_row['val_loss']:.4f}")
    print(f"    Copied checkpoint to: {copy_to}")
    print(f"    Full table saved to : {run_dir / 'posthoc_train_val_summary.csv'}")

if __name__ == "__main__":
    main()
