import argparse, glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------- style (journal friendly) ----------
mpl.rcParams.update({
    "figure.dpi": 180,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.frameon": False,
})

def ema(x, alpha):
    if alpha is None or alpha <= 0:
        return np.asarray(x, dtype=float)
    y = np.zeros_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1.0 - alpha) * y[i-1]
    return y

def _read_epochs_csv(csv_path):
    df = pd.read_csv(csv_path)
    # tolerant column names
    rename_map = {
        "epoch": "epoch",
        "epochs": "epoch",
        "train_acc": "train_acc",
        "tr_acc": "train_acc",
        "train_accuracy": "train_acc",
        "val_acc": "val_acc",
        "valid_acc": "val_acc",
        "validation_acc": "val_acc",
        "train_loss": "train_loss",
        "tr_loss": "train_loss",
        "val_loss": "val_loss",
        "valid_loss": "val_loss",
        "validation_loss": "val_loss",
    }
    cols = {c: rename_map.get(c, c) for c in df.columns}
    df = df.rename(columns=cols)
    needed = ["train_acc", "val_acc", "train_loss", "val_loss"]
    for k in needed:
        if k not in df.columns:
            raise ValueError(f"{csv_path} missing column: {k}")
    # epoch column (1..E) if not present
    if "epoch" not in df.columns:
        df["epoch"] = np.arange(1, len(df) + 1)
    return df

def plot_fold_panel(fold_name, df, out_path, ema_alpha=None):
    x = df["epoch"].to_numpy()
    tr_acc = ema(df["train_acc"].to_numpy(), ema_alpha)
    va_acc = ema(df["val_acc"].to_numpy(), ema_alpha)
    tr_loss = ema(df["train_loss"].to_numpy(), ema_alpha)
    va_loss = ema(df["val_loss"].to_numpy(), ema_alpha)

    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.4), constrained_layout=True)

    # Accuracy
    ax[0].plot(x, tr_acc, label="Train", lw=2)
    ax[0].plot(x, va_acc, label="Val", lw=2)
    ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Accuracy")
    ax[0].set_ylim(0.0, 1.02)
    ax[0].set_title(f"{fold_name} — Training vs Validation Accuracy")
    ax[0].legend()

    # Loss
    ax[1].plot(x, tr_loss, label="Train", lw=2)
    ax[1].plot(x, va_loss, label="Val", lw=2)
    ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Loss")
    ax[1].set_title(f"{fold_name} — Training vs Validation Loss")
    ax[1].legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"))
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)

def overlay_all(fold_series, out_png, ylabel, title):
    plt.figure(figsize=(7.2, 4.0))
    for fold, (x, y) in fold_series.items():
        plt.plot(x, y, lw=2, label=fold)
    plt.xlabel("Epoch"); plt.ylabel(ylabel); plt.title(title)
    if ylabel.lower().startswith("acc"):
        plt.ylim(0.0, 1.02)
    plt.legend(ncol=2)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.savefig(Path(out_png).with_suffix(".pdf"))
    plt.close()

def grid_all(folds, dfs, out_png, ema_alpha=None):
    # 5 folds -> 2x3 grid; last empty panel hidden
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.8), constrained_layout=True)
    axes = axes.ravel()
    for i, (fold, df) in enumerate(zip(folds, dfs)):
        ax = axes[i]
        x = df["epoch"].to_numpy()
        tr = ema(df["train_acc"].to_numpy(), ema_alpha)
        va = ema(df["val_acc"].to_numpy(), ema_alpha)
        ax.plot(x, tr, lw=1.8, label="Train")
        ax.plot(x, va, lw=1.8, label="Val")
        ax.set_title(f"{fold} — Accuracy")
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Acc")
        if i == 0: ax.legend(frameon=False)
    for j in range(len(dfs), len(axes)):  # hide extras
        axes[j].axis("off")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png); fig.savefig(Path(out_png).with_suffix(".pdf"))
    plt.close(fig)

    # Loss grid
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.8), constrained_layout=True)
    axes = axes.ravel()
    for i, (fold, df) in enumerate(zip(folds, dfs)):
        ax = axes[i]
        x = df["epoch"].to_numpy()
        tr = ema(df["train_loss"].to_numpy(), ema_alpha)
        va = ema(df["val_loss"].to_numpy(), ema_alpha)
        ax.plot(x, tr, lw=1.8, label="Train")
        ax.plot(x, va, lw=1.8, label="Val")
        ax.set_title(f"{fold} — Loss")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        if i == 0: ax.legend(frameon=False)
    for j in range(len(dfs), len(axes)):
        axes[j].axis("off")
    fig.savefig(Path(out_png).with_name(Path(out_png).stem.replace("acc", "loss") + ".png"))
    fig.savefig(Path(out_png).with_name(Path(out_png).stem.replace("acc", "loss") + ".pdf"))
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="cv run dir containing fold_*/epochs.csv")
    ap.add_argument("--ema", type=float, default=0.0, help="EMA factor (0 disables)")
    ap.add_argument("--outdir", default=None, help="optional output dir (defaults to run_dir)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.outdir) if args.outdir else run_dir

    # find folds
    csvs = sorted(run_dir.glob("fold_*/epochs.csv"))
    if not csvs:
        raise SystemExit(f"No epochs.csv found under {run_dir}/fold_*/")

    folds, dfs = [], []
    # per-fold panels
    for csv_path in csvs:
        fold_name = csv_path.parent.name.replace("_", " ").title()  # e.g., 'Fold 01'
        df = _read_epochs_csv(csv_path)
        folds.append(fold_name); dfs.append(df)
        out_path = out_dir / f"fig_trainval_panel_{csv_path.parent.name}"
        plot_fold_panel(fold_name, df, out_path, ema_alpha=args.ema)

    # overlays
    acc_series = {f: (d["epoch"].to_numpy(),
                      ema(d["val_acc"].to_numpy(), args.ema)) for f, d in zip(folds, dfs)}
    loss_series = {f: (d["epoch"].to_numpy(),
                       ema(d["val_loss"].to_numpy(), args.ema)) for f, d in zip(folds, dfs)}
    overlay_all(acc_series, out_dir / "fig_val_acc_overlay.png",
                ylabel="Accuracy", title="Validation Accuracy Across Folds")
    overlay_all(loss_series, out_dir / "fig_val_loss_overlay.png",
                ylabel="Loss", title="Validation Loss Across Folds")

    # grids
    grid_all(folds, dfs, out_dir / "fig_trainval_acc_grid.png", ema_alpha=args.ema)

    print(f"[OK] Saved per-fold panels, overlays, and grids under: {out_dir}")

if __name__ == "__main__":
    main()


# # scripts/make_curves_from_epochs.py
# import argparse
# from pathlib import Path
# import numpy as np, pandas as pd, matplotlib.pyplot as plt
# plt.rcParams.update({
#     "figure.dpi": 180, "savefig.dpi": 300,
#     "axes.spines.top": False, "axes.spines.right": False,
#     "font.size": 12, "legend.frameon": False
# })
#
# def _load_epochs(run_dir: Path):
#     files = sorted(run_dir.glob("fold_*/epochs.csv"))
#     if not files:
#         raise SystemExit(f"No epochs.csv in {run_dir}")
#     tr_loss, tr_acc, va_loss, va_acc = [], [], [], []
#     Lmin = 10**9
#     for f in files:
#         df = pd.read_csv(f)
#         for col in ["train_loss","train_acc","val_loss","val_acc"]:
#             if col not in df: raise SystemExit(f"Missing column {col} in {f}")
#         Lmin = min(Lmin, len(df))
#         tr_loss.append(df["train_loss"].to_numpy())
#         tr_acc.append(df["train_acc"].to_numpy())
#         va_loss.append(df["val_loss"].to_numpy())
#         va_acc.append(df["val_acc"].to_numpy())
#     # align lengths (handles early stop)
#     tr_loss = np.vstack([a[:Lmin] for a in tr_loss])
#     tr_acc  = np.vstack([a[:Lmin] for a in tr_acc])
#     va_loss = np.vstack([a[:Lmin] for a in va_loss])
#     va_acc  = np.vstack([a[:Lmin] for a in va_acc])
#     return tr_loss, tr_acc, va_loss, va_acc
#
# def _ema(arr2d, alpha):
#     if alpha <= 0: return arr2d
#     out = np.empty_like(arr2d)
#     for i in range(arr2d.shape[0]):
#         x = arr2d[i]
#         y = np.empty_like(x); y[0] = x[0]
#         for t in range(1, len(x)):
#             y[t] = alpha * x[t] + (1 - alpha) * y[t-1]
#         out[i] = y
#     return out
#
# def _plot_meanstd(ax, x, mats, label):
#     m = np.nanmean(mats, axis=0)
#     s = np.nanstd(mats, axis=0, ddof=1) if mats.shape[0] > 1 else np.zeros_like(m)
#     line, = ax.plot(x, m, lw=2.6, label=label)
#     ax.fill_between(x, m - s, m + s, alpha=0.18, color=line.get_color())
#
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--run_dir", required=True, help="artifacts/cv/cvtest_..._5f")
#     ap.add_argument("--ema", type=float, default=0.15, help="EMA smoothing factor (0=off)")
#     args = ap.parse_args()
#
#     run_dir = Path(args.run_dir)
#     tr_loss, tr_acc, va_loss, va_acc = _load_epochs(run_dir)
#
#     # light smoothing for nicer publication curves
#     tr_loss = _ema(tr_loss, args.ema)
#     tr_acc  = _ema(tr_acc,  args.ema)
#     va_loss = _ema(va_loss, args.ema)
#     va_acc  = _ema(va_acc,  args.ema)
#
#     x = np.arange(1, tr_loss.shape[1] + 1)
#     fig, ax = plt.subplots(1,2, figsize=(14,4.8), constrained_layout=True)
#
#     _plot_meanstd(ax[0], x, tr_acc, "Train")
#     _plot_meanstd(ax[0], x, va_acc, "Val")
#     ax[0].set_title("Training vs Validation Accuracy")
#     ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Accuracy"); ax[0].grid(alpha=0.15)
#
#     _plot_meanstd(ax[1], x, tr_loss, "Train")
#     _plot_meanstd(ax[1], x, va_loss, "Val")
#     ax[1].set_title("Training vs Validation Loss")
#     ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Loss"); ax[1].grid(alpha=0.15)
#     ax[1].legend(loc="upper right")
#
#     png = run_dir / "fig_trainval_panel_epochs.png"
#     pdf = run_dir / "fig_trainval_panel_epochs.pdf"
#     fig.savefig(png); fig.savefig(pdf)
#     print(f"[OK] saved {png}\n[OK] saved {pdf}")
#
# if __name__ == "__main__":
#     main()
