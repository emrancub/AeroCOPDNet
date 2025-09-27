# scripts/augment_viz.py
import argparse, os
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
import matplotlib.pyplot as plt

from src.copd.features import build_feature
from src.copd.augment import WaveAugment, SpecAugment

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

def to_numpy_spec(spec_1_1_F_T):
    # (1,1,F,T) -> (F,T) numpy
    return spec_1_1_F_T.squeeze(0).squeeze(0).detach().cpu().numpy()

def plot_row(ax_wav, ax_spec, wav_1_L, spec_1_1_F_T, sr, title_wav, title_spec):
    t = torch.arange(wav_1_L.shape[-1]) / sr
    ax_wav.plot(t.numpy(), wav_1_L.squeeze(0).detach().cpu().numpy())
    ax_wav.set_title(title_wav); ax_wav.set_xlabel("Time (s)"); ax_wav.set_ylabel("Amp")
    S = to_numpy_spec(spec_1_1_F_T)
    im = ax_spec.imshow(S, aspect="auto", origin="lower")
    ax_spec.set_title(title_spec); ax_spec.set_xlabel("Frames"); ax_spec.set_ylabel("Mel bins")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True, help="output PNG")
    ap.add_argument("--n", type=int, default=6)  # not used heavily, just for CLI compatibility
    ap.add_argument("--features", default="mel")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--use_waveaug", action="store_true")
    ap.add_argument("--use_specaug", action="store_true")
    ap.add_argument("--mixup", type=float, default=0.0)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if not {"path", "label"}.issubset(df.columns):
        raise ValueError("CSV must have columns: path,label[,...]")

    # choose one example (first row)
    p0 = df.iloc[0]["path"]
    wav0 = load_wav(p0, args.sr, args.duration)         # (1,L)

    # build feature (mel or mfcc)
    feat = build_feature(args.features, sample_rate=args.sr)

    # augmentation modules
    wave_gain   = WaveAugment(sample_rate=args.sr, p_gain=1.0, p_noise=0.0, p_shift=0.0, p_stretch=0.0)
    wave_noise  = WaveAugment(sample_rate=args.sr, p_gain=0.0, p_noise=1.0, p_shift=0.0, p_stretch=0.0)
    wave_shift  = WaveAugment(sample_rate=args.sr, p_gain=0.0, p_noise=0.0, p_shift=1.0, p_stretch=1.0)
    specaug     = SpecAugment(p=1.0) if args.use_specaug else None

    rows = 5 + (1 if args.mixup > 0 else 0)
    fig, axes = plt.subplots(rows, 2, figsize=(10, 2.6 * rows), constrained_layout=True)

    r = 0
    # 1) Original
    spec0 = feat(wav0)  # (1,1,F,T)
    plot_row(axes[r,0], axes[r,1], wav0, spec0, args.sr, "Original", "Original (Log-mel)")
    r += 1

    # 2) Random Gain
    if args.use_waveaug:
        wav_g = wave_gain(wav0)
        spec_g = feat(wav_g)
    else:
        wav_g, spec_g = wav0, spec0
    plot_row(axes[r,0], axes[r,1], wav_g, spec_g, args.sr, "Random Gain", "Random Gain (Log-mel)")
    r += 1

    # 3) Add Noise
    if args.use_waveaug:
        wav_n = wave_noise(wav0)
        spec_n = feat(wav_n)
    else:
        wav_n, spec_n = wav0, spec0
    plot_row(axes[r,0], axes[r,1], wav_n, spec_n, args.sr, "Add Noise", "Add Noise (Log-mel)")
    r += 1

    # 4) Time Shift / Stretch
    if args.use_waveaug:
        wav_s = wave_shift(wav0)
        spec_s = feat(wav_s)
    else:
        wav_s, spec_s = wav0, spec0
    plot_row(axes[r,0], axes[r,1], wav_s, spec_s, args.sr, "Time Shift / Stretch", "Time Shift / Stretch (Log-mel)")
    r += 1

    # 5) SpecAugment (applied to spectrogram)
    if specaug is not None:
        spec_sa = specaug(spec0)  # same rank (1,1,F,T)
    else:
        spec_sa = spec0
    plot_row(axes[r,0], axes[r,1], wav0, spec_sa, args.sr, "Original (for SpecAug)", "SpecAugment (Log-mel)")
    r += 1

    # 6) Optional MixUp row
    if args.mixup > 0:
        # pick a different second file
        p1 = df.iloc[min(len(df)-1, 1)]["path"]
        wav1 = load_wav(p1, args.sr, args.duration)
        import numpy as np
        lam = float(np.random.beta(args.mixup, args.mixup))
        wav_m = lam * wav0 + (1.0 - lam) * wav1
        spec_m = feat(wav_m)
        plot_row(axes[r,0], axes[r,1], wav_m, spec_m, args.sr, f"MixUp (λ={lam:.2f})", "MixUp (Log-mel)")

    out = Path(args.out)
    ensure_dir(out)
    fig.savefig(out, dpi=300)
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
