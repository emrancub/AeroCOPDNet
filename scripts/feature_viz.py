# scripts/feature_viz.py
import argparse
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
import matplotlib.pyplot as plt

from src.copd.features import build_feature

def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def load_wav(path, sr, duration_s):
    wav, sr0 = torchaudio.load(path)         # (C,L)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)  # mono (1,L)
    if sr0 != sr:
        wav = torchaudio.functional.resample(wav, sr0, sr)
    target = int(sr * duration_s)
    L = wav.shape[-1]
    if L < target:
        wav = F.pad(wav, (0, target - L))
    elif L > target:
        wav = wav[..., :target]
    return wav  # (1,L)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True, help="output PNG")
    ap.add_argument("--n", type=int, default=8)             # number of examples to show
    ap.add_argument("--features", nargs="+", default=["mel","mfcc"])
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--duration", type=float, default=4.0)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if not {"path", "label"}.issubset(df.columns):
        raise ValueError("CSV must have columns: path,label[,...]")

    paths = df["path"].tolist()[:args.n]
    feats = {name: build_feature(name, sample_rate=args.sr) for name in args.features}

    n_rows = 1 + len(feats)            # waveform row + one row per feature
    n_cols = args.n
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6*n_cols, 2.2*n_rows), constrained_layout=True)

    # 1) waveforms
    for j, p in enumerate(paths):
        wav = load_wav(p, args.sr, args.duration)
        t = torch.arange(wav.shape[-1]) / args.sr
        ax = axes[0, j] if n_cols > 1 else axes[0]
        ax.plot(t.numpy(), wav.squeeze(0).numpy())
        ax.set_title(Path(p).name[:25])
        ax.set_xticks([]); ax.set_yticks([])

        # compute and cache features
        for i, (fname, fmod) in enumerate(feats.items(), start=1):
            spec = fmod(wav)                                 # (1,1,F,T)
            S = spec.squeeze(0).squeeze(0).numpy()          # (F,T)
            axf = axes[i, j] if n_cols > 1 else axes[i]
            axf.imshow(S, aspect="auto", origin="lower")
            if j == 0: axf.set_ylabel(fname.upper())
            axf.set_xticks([]); axf.set_yticks([])

    out = Path(args.out)
    ensure_dir(out)
    fig.savefig(out, dpi=300)
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
