# scripts/train.py
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, confusion_matrix,
    accuracy_score, precision_recall_curve, roc_curve, balanced_accuracy_score
)

from src.copd.utils import ensure_dir, set_seed, device_str, timestamp
from src.copd.data import AudioBinaryDataset, build_sampler
from src.copd.augment import SpecAugment
from src.copd.features import build_feature
from src.copd.old_models import build_model
from src.copd.trainloop import train_one_epoch, evaluate


def _infer_test_csv(train_csv: Path) -> Path | None:
    p = Path(train_csv)
    cands = []
    if p.stem.endswith("_train"):
        cands.append(p.with_name(p.stem.replace("_train", "_test") + p.suffix))
    cands.append(p.with_name(p.stem + "_test.csv"))
    for c in cands:
        if c.exists():
            return c
    return None


def _plot_curves(df: pd.DataFrame, out_png: Path):
    ensure_dir(out_png.parent)
    plt.figure(figsize=(9,4.5))
    # Loss
    plt.subplot(1,2,1)
    plt.plot(df["epoch"], df["train_loss"], label="train")
    plt.plot(df["epoch"], df["val_loss"],   label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Loss"); plt.legend()
    # Accuracy
    plt.subplot(1,2,2)
    plt.plot(df["epoch"], df["train_acc"], label="train")
    plt.plot(df["epoch"], df["val_acc"],   label="val")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.title("Accuracy"); plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


@torch.no_grad()
def _predict_loader(model, loader, feat, device):
    """Return y_true, y_prob, y_pred, avg_loss."""
    model.eval()
    all_probs, all_true = [], []
    bce = torch.nn.BCEWithLogitsLoss(reduction="sum")
    total_loss, total_n = 0.0, 0
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            if len(batch) == 3:
                X, y, _ = batch
            else:
                X, y = batch
        else:
            X, y = batch["wave"], batch["label"]
        Xf = feat(X).to(device)     # (B,1,F,T)
        y   = torch.as_tensor(y, device=device).float().view(-1,1)
        logit = model(Xf)
        loss = bce(logit, y)
        prob = torch.sigmoid(logit).view(-1).detach().cpu().numpy()
        all_probs.append(prob)
        all_true.append(y.view(-1).detach().cpu().numpy())
        total_loss += float(loss.item())
        total_n += y.numel()
    y_true = np.concatenate(all_true) if all_true else np.array([])
    y_prob = np.concatenate(all_probs) if all_probs else np.array([])
    y_pred = (y_prob >= 0.5).astype(int)
    avg_loss = total_loss / max(1, total_n)
    return y_true, y_prob, y_pred, avg_loss


def _test_report(y_true, y_prob, y_pred):
    acc  = accuracy_score(y_true, y_pred)
    auc  = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    aupr = average_precision_score(y_true, y_prob)
    f1   = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    bal  = balanced_accuracy_score(y_true, y_pred)
    return dict(accuracy=acc, auc_roc=auc, aupr=aupr, f1=f1,
                sensitivity=sens, specificity=spec, balanced_accuracy=bal,
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def _plot_roc_pr_confmat(y_true, y_prob, out_prefix: Path):
    ensure_dir(out_prefix.parent)

    # ROC
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure()
        plt.plot(fpr, tpr, label="ROC")
        plt.plot([0,1],[0,1], linestyle="--")
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC")
        plt.legend(); plt.tight_layout()
        plt.savefig(str(out_prefix) + "_roc.png", dpi=160)
        plt.close()
    except Exception:
        pass

    # PR
    try:
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        plt.figure()
        plt.plot(rec, prec, label="PR")
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall")
        plt.legend(); plt.tight_layout()
        plt.savefig(str(out_prefix) + "_pr.png", dpi=160)
        plt.close()
    except Exception:
        pass

    # Confusion matrix
    cm = confusion_matrix(y_true, (y_prob>=0.5).astype(int))
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix"); plt.colorbar()
    plt.xticks([0,1], ["Non-COPD","COPD"])
    plt.yticks([0,1], ["Non-COPD","COPD"])
    for (i,j),v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(str(out_prefix) + "_confmat.png", dpi=160)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Labels CSV (train or pooled)")
    ap.add_argument("--csv_test", default=None, help="Optional explicit test CSV")
    ap.add_argument("--model", default="aerocpdnet",
                    choices=["basiccnn","crnn","lstm","bilstm","gru","aerocpdnet"])
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

    # class balance handling
    tr_labels = np.array([ds[i][1] for i in tr_ds.indices], dtype=int)
    if args.no_class_weights:
        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
        pos_weight = None
    else:
        sampler = build_sampler(tr_labels)
        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0)
        c = np.bincount(tr_labels, minlength=2)
        pos_weight = c[0] / max(1, c[1])

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # features / augment
    feat = build_feature(args.features)
    specaug = SpecAugment() if args.use_specaug else None

    # model/opt
    model = build_model(args.model, in_ch=1, n_classes=1, dropout=args.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_id = f"{args.model}_{timestamp()}"
    logs_dir = Path("artifacts/logs"); ensure_dir(logs_dir)
    models_dir = Path("artifacts/models"); ensure_dir(models_dir)
    plots_dir = Path("artifacts/plots"); ensure_dir(plots_dir)

    out_csv = logs_dir / f"epochs_{run_id}.csv"
    out_ckpt = models_dir / f"{run_id}_best.pt"

    best_score = -1.0
    rows = []

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, tr_loader, opt, feat, specaug, dev, pos_weight, mixup_alpha=args.mixup
        )
        val_stats = evaluate(model, val_loader, feat, device=dev)

        acc_val = float(val_stats.get("accuracy", val_stats.get("acc", float("nan"))))
        auc_val = float(val_stats.get("auc_roc", val_stats.get("auc", float("nan"))))
        aupr_val = float(val_stats.get("aupr", float("nan")))
        f1_val   = float(val_stats.get("f1", float("nan")))
        sens_val = float(val_stats.get("sensitivity", float("nan")))
        spec_val = float(val_stats.get("specificity", float("nan")))
        bal_val  = float(val_stats.get("balanced_accuracy", float("nan")))
        loss_val = float(val_stats.get("loss", float("nan")))

        rows.append(dict(
            epoch=epoch, train_loss=tr_loss, train_acc=tr_acc,
            val_loss=loss_val, val_acc=acc_val, val_auc=auc_val,
            val_aupr=aupr_val, val_f1=f1_val, val_sens=sens_val,
            val_spec=spec_val, val_bal_acc=bal_val
        ))

        print(f"Epoch {epoch:03d} | "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.3f} | "
              f"val_loss={loss_val:.4f} val_acc={acc_val:.3f} "
              f"AUC={auc_val:.4f} AUPR={aupr_val:.4f} F1={f1_val:.3f}")

        # choose best by AUC if available, else balanced acc, else acc
        score = auc_val if not np.isnan(auc_val) else (bal_val if not np.isnan(bal_val) else acc_val)
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), out_ckpt)

    # write epoch CSV + curves
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    _plot_curves(df, plots_dir / f"curves_{run_id}.png")
    print(f"[saved] {out_csv}")
    print(f"[saved] {plots_dir / f'curves_{run_id}.png'}")
    print(f"[best ckpt] {out_ckpt}")

    # ------------------ TEST EVAL ------------------
    test_csv = Path(args.csv_test) if args.csv_test else _infer_test_csv(Path(args.csv))
    if test_csv is None or not test_csv.exists():
        print("[WARN] No test CSV found. Pass --csv_test to compute test metrics/plots.")
        return

    print(f"[TEST] Using: {test_csv}")
    # load best checkpoint
    state = torch.load(out_ckpt, map_location=device_str())
    model.load_state_dict(state)

    test_ds = AudioBinaryDataset(test_csv, sample_rate=args.sr, duration=args.duration, return_path=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    y_true, y_prob, y_pred, test_loss = _predict_loader(model, test_loader, feat, dev)
    test_metrics = _test_report(y_true, y_prob, y_pred)
    test_metrics["loss"] = float(test_loss)

    out_prefix = logs_dir / f"{test_csv.stem}_{run_id}"
    _plot_roc_pr_confmat(y_true, y_prob, out_prefix)

    # preds CSV
    preds_csv = str(out_prefix) + "_preds.csv"
    pd.DataFrame({"path":[p for p,_,__ in test_ds.rows] if hasattr(test_ds, "rows") else None,
                  "y_true": y_true, "y_prob": y_prob, "y_pred": y_pred}).to_csv(preds_csv, index=False)

    # metrics JSON
    metrics_json = str(out_prefix) + "_metrics.json"
    Path(metrics_json).write_text(json.dumps(test_metrics, indent=2))
    print(f"[TEST] loss={test_metrics['loss']:.4f} acc={test_metrics['accuracy']:.3f} "
          f"AUC={test_metrics['auc_roc']:.4f} AUPR={test_metrics['aupr']:.4f} "
          f"F1={test_metrics['f1']:.3f} sens={test_metrics['sensitivity']:.3f} "
          f"spec={test_metrics['specificity']:.3f} bal_acc={test_metrics['balanced_accuracy']:.3f}")
    print(f"[saved] {preds_csv}")
    print(f"[saved] {metrics_json}")
    print(f"[saved] {str(out_prefix) + '_roc.png'}")
    print(f"[saved] {str(out_prefix) + '_pr.png'}")
    print(f"[saved] {str(out_prefix) + '_confmat.png'}")


if __name__ == "__main__":
    main()
