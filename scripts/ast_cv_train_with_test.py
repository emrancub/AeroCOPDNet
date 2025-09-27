# scripts/cv_train_with_test.py
import argparse, os, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold

from src.copd.utils import ensure_dir, set_seed, device_str, timestamp
from src.copd.data import AudioBinaryDataset, build_sampler
from src.copd.features import build_feature
from src.copd.ast_models import build_model
from src.copd.trainloop import train_one_epoch, evaluate


# -----------------------------
# helpers
# -----------------------------
def stratified_split(df, test_size=0.15, seed=42):
    """Return df_trainval, df_test (stratified by label; patient-wise if patient_id exists)."""
    df = df.copy()
    if "label" not in df.columns:
        raise ValueError("CSV must contain 'label' column.")
    y = df["label"].values.astype(int)

    # Patient-wise holdout if available
    if "patient_id" in df.columns:
        rng = np.random.RandomState(seed)
        pats = df["patient_id"].astype(str).unique()
        rng.shuffle(pats)
        n_test_pat = max(1, int(round(len(pats) * test_size)))
        test_pats = set(pats[:n_test_pat])
        is_test = df["patient_id"].astype(str).isin(test_pats).values
    else:
        n = len(df)
        idx = np.arange(n)
        n_splits = max(2, int(round(1.0 / test_size)))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        tr_idx, te_idx = next(iter(skf.split(idx, y)))
        is_test = np.zeros(n, dtype=bool)
        is_test[te_idx] = True

    df_test = df[is_test].reset_index(drop=True)
    df_trainval = df[~is_test].reset_index(drop=True)
    return df_trainval, df_test


def metrics_get(d, key, default=np.nan):
    # robustness for slightly different naming
    return float(d.get(key, d.get(key.replace("auc_roc", "auc"), default)))


# -----------------------------
# fold runner
# -----------------------------
def run_fold(fold, tr_idx, va_idx, base_ds, args, dev, run_dir, feat, specaug):
    fold_dir = Path(run_dir) / f"fold_{fold:02d}"
    ensure_dir(fold_dir)

    # ---- datasets & loaders
    tr_ds = Subset(base_ds, tr_idx)
    va_ds = Subset(base_ds, va_idx)

    # sampler / class-weights
    tr_labels = np.array([base_ds[i][1] for i in tr_idx], dtype=int)
    if args.no_class_weights:
        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=0, drop_last=False)
        pos_weight = None
    else:
        sampler = build_sampler(tr_labels)
        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, sampler=sampler,
                               num_workers=0, drop_last=False)
        c = np.bincount(tr_labels, minlength=2)
        pos_weight = (c[0] / max(1, c[1])).astype(float)

    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=0, drop_last=False)

    # ---- model & optim (pretrained with fallback)
    try:
        model = build_model(args.model, in_ch=1, n_classes=1, dropout=args.dropout, pretrained=True).to(dev)
    except Exception as e:
        print(f"[WARN] Pretrained build failed ({e}). Falling back to pretrained=False.")
        model = build_model(args.model, in_ch=1, n_classes=1, dropout=args.dropout, pretrained=False).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_score = -1.0
    best_ckpt_path = fold_dir / "best.pt"

    rows = []
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, tr_loader, opt, feat, specaug, dev,
            pos_weight=pos_weight, mixup_alpha=args.mixup
        )
        val_stats = evaluate(model, va_loader, feat, device=dev)

        # robust keys
        val_loss = metrics_get(val_stats, "loss")
        val_acc  = metrics_get(val_stats, "accuracy", val_stats.get("acc", np.nan))
        val_auc  = metrics_get(val_stats, "auc_roc", val_stats.get("auc", np.nan))
        val_aupr = metrics_get(val_stats, "aupr", val_stats.get("average_precision", np.nan))
        val_f1   = metrics_get(val_stats, "f1")

        rows.append(dict(
            epoch=epoch, train_loss=tr_loss, train_acc=tr_acc,
            val_loss=val_loss, val_acc=val_acc, val_auc=val_auc, val_aupr=val_aupr, val_f1=val_f1
        ))

        score = val_auc if not np.isnan(val_auc) else (val_stats.get("f1", -1.0))
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), best_ckpt_path)

        print(f"[Fold {fold}] Epoch {epoch:03d} | tr_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
              f"AUC={val_auc:.4f} AUPR={val_aupr:.4f} F1={val_f1:.3f} acc={val_acc:.3f}")

    # Save per-epoch val history
    pd.DataFrame(rows).to_csv(fold_dir / "epochs.csv", index=False)

    last_row = rows[-1] if rows else {}
    return best_ckpt_path, last_row


# -----------------------------
# test evaluation (no downloads)
# -----------------------------
@torch.no_grad()
def eval_on_test(best_ckpt_path: Path, args, dev, feat, test_loader):
    # Safe: never allow HF download at test-time
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    model = build_model(args.model, in_ch=1, n_classes=1, dropout=args.dropout, pretrained=False).to(dev).eval()

    state = torch.load(best_ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # forgiving strict=False load
    model.load_state_dict(state, strict=False)

    stats = evaluate(model, test_loader, feat, device=dev)
    stats["checkpoint"] = str(best_ckpt_path)

    # roll-up (standard names)
    out = dict(
        loss = metrics_get(stats, "loss"),
        acc  = metrics_get(stats, "accuracy", stats.get("acc", np.nan)),
        auc  = metrics_get(stats, "auc_roc", stats.get("auc", np.nan)),
        aupr = metrics_get(stats, "aupr", stats.get("average_precision", np.nan)),
        f1   = metrics_get(stats, "f1"),
        prec = metrics_get(stats, "precision"),
        recall = metrics_get(stats, "recall"),
        sens = metrics_get(stats, "sensitivity"),
        spec = metrics_get(stats, "specificity"),
        bal_acc = metrics_get(stats, "balanced_accuracy"),
        mcc  = metrics_get(stats, "mcc"),
        brier= metrics_get(stats, "brier"),
        checkpoint = str(best_ckpt_path),
    )
    # optional curves / predictions if present
    for k in ("y_true","y_prob","fpr","tpr","prec_curve","rec_curve"):
        if k in stats:
            v = np.asarray(stats[k])
            out[k] = v.tolist()
    return out


# -----------------------------
# main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--csv_test", default=None, help="Optional separate CSV for test set.")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--features", default="mel")
    ap.add_argument("--model", default="aerocpdnet")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--no_class_weights", action="store_true")
    ap.add_argument("--use_specaug", action="store_true")
    ap.add_argument("--mixup", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--test_size", type=float, default=0.15)
    args = ap.parse_args()

    set_seed(args.seed)
    dev = device_str()
    print(f"Device: {dev}")

    run_dir = Path(f"artifacts/cv/cvtest_{timestamp()}_{args.folds}f")
    ensure_dir(run_dir)

    # Read master CSV
    df_all = pd.read_csv(args.csv)

    # Build test split
    if args.csv_test:
        df_test = pd.read_csv(args.csv_test)
        df_pool = df_all.copy()
        print(f"Using explicit test CSV: {len(df_test)} rows.")
    else:
        df_pool, df_test = stratified_split(df_all, test_size=args.test_size, seed=args.seed)
        print(f"Auto-split test set: {len(df_test)} rows; train+val pool: {len(df_pool)} rows.")

    # Save splits for reproducibility
    test_csv_path = run_dir / "test_split.csv"
    df_test.to_csv(test_csv_path, index=False)
    pool_csv_path = run_dir / "trainval_pool.csv"
    df_pool.to_csv(pool_csv_path, index=False)

    # Datasets
    base_ds = AudioBinaryDataset(str(pool_csv_path), sample_rate=args.sr, duration=args.duration, return_path=True)
    test_ds = AudioBinaryDataset(str(test_csv_path), sample_rate=args.sr, duration=args.duration, return_path=True)

    # Feature & SpecAug
    feat = build_feature(args.features)  # CPU transform; trainloop moves to GPU
    specaug = None
    if args.use_specaug:
        from src.copd.augment import SpecAugment
        specaug = SpecAugment()

    # K-fold on pool (stratified by label)
    y_pool = df_pool["label"].astype(int).values
    idx_pool = np.arange(len(df_pool))
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    # Shared test loader
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    last_epoch_rows = []
    test_rows = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(idx_pool, y_pool), start=1):
        print(f"Fold {fold}/{args.folds}: train={len(tr_idx)}, val={len(va_idx)}")

        best_ckpt_path, last_row = run_fold(fold, tr_idx, va_idx, base_ds, args, dev, run_dir, feat, specaug)
        last_epoch_rows.append({"fold": fold, **last_row})

        # Evaluate best checkpoint on the constant test set
        fold_dir = run_dir / f"fold_{fold:02d}"
        ensure_dir(fold_dir)

        test_stats = eval_on_test(best_ckpt_path, args, dev, feat, test_loader)
        with open(fold_dir / "test_metrics.json", "w") as f:
            json.dump(test_stats, f, indent=2)

        if "y_true" in test_stats and "y_prob" in test_stats:
            pd.DataFrame({"y_true": test_stats["y_true"], "y_prob": test_stats["y_prob"]}).to_csv(
                fold_dir / "test_predictions.csv", index=False
            )

        test_rows.append({
            "fold": fold,
            "test_loss": test_stats.get("loss", np.nan),
            "test_acc": test_stats.get("acc", np.nan),
            "test_auc": test_stats.get("auc", np.nan),
            "test_aupr": test_stats.get("aupr", np.nan),
            "test_f1": test_stats.get("f1", np.nan),
            "test_prec": test_stats.get("prec", np.nan),
            "test_recall": test_stats.get("recall", np.nan),
            "test_sens": test_stats.get("sens", np.nan),
            "test_spec": test_stats.get("spec", np.nan),
            "test_bal_acc": test_stats.get("bal_acc", np.nan),
            "test_mcc": test_stats.get("mcc", np.nan),
            "test_brier": test_stats.get("brier", np.nan),
        })

    # Save per-fold last-epoch validation summary
    df_val_last = pd.DataFrame(last_epoch_rows)
    df_val_last.to_csv(run_dir / "val_last_epoch_summary.csv", index=False)

    # Save per-fold test metrics & mean±std
    df_testall = pd.DataFrame(test_rows)
    df_testall.to_csv(run_dir / "test_per_fold.csv", index=False)

    # Aggregate
    agg = {}
    keys = ["test_loss","test_acc","test_auc","test_aupr","test_f1","test_prec","test_recall",
            "test_sens","test_spec","test_bal_acc","test_mcc","test_brier"]
    for k in keys:
        vals = df_testall[k].values.astype(float)
        agg[f"{k}_mean"] = float(np.nanmean(vals))
        agg[f"{k}_std"]  = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0

    with open(run_dir / "test_summary_mean_std.json", "w") as f:
        json.dump(agg, f, indent=2)

    print("====== FINAL TEST SUMMARY (mean ± std) ======")
    for k in ["acc","auc","aupr","f1","prec","recall","sens","spec","bal_acc","mcc","brier","loss"]:
        m = agg.get(f"test_{k}_mean", np.nan)
        s = agg.get(f"test_{k}_std", np.nan)
        print(f"{k:8s}: {m:.4f} ± {s:.4f}")

    print(f"All artifacts in: {run_dir}")


if __name__ == "__main__":
    main()
