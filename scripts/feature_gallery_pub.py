# scripts/feature_gallery_pub.py
import os
# Avoid the Intel OpenMP duplicate crash on Windows when importing librosa
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_audio(path: str, sr: int, duration_s: float):
    """Mono, fixed length."""
    y, sr0 = librosa.load(path, sr=None, mono=True)
    if sr0 != sr:
        y = librosa.resample(y, orig_sr=sr0, target_sr=sr)
    target = int(sr * duration_s)
    if y.shape[0] < target:
        y = np.pad(y, (0, target - y.shape[0]))
    else:
        y = y[:target]
    return y, sr

def compute_features(y, sr, n_fft=1024, hop=256, n_mels=64, n_mfcc=20):
    # Log-mel
    mel = librosa.power_to_db(
        librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels),
        ref=np.max
    )
    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop, n_fft=n_fft)
    # Chroma & Spectral Contrast
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    # Temporal features (frame-wise)
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop).squeeze(0)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop).squeeze(0)
    return dict(mel=mel, mfcc=mfcc, chroma=chroma, contrast=contrast, rms=rms, zcr=zcr, hop=hop)

def plot_gallery(y, sr, feats, title_prefix, out_png):
    T = np.arange(y.shape[0]) / sr
    hop = feats["hop"]
    frames = np.arange(feats["mel"].shape[1])

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)

    # Waveform
    ax = axes[0, 0]
    ax.plot(T, y, linewidth=1.0)
    ax.set_title(f"{title_prefix} - Waveform")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Amp")

    # Log-mel
    ax = axes[0, 1]
    im = ax.imshow(feats["mel"], aspect="auto", origin="lower", cmap="magma")
    ax.set_title("Log-mel Spectrogram"); ax.set_xlabel("Frames"); ax.set_ylabel("Mel bins")

    # MFCC
    ax = axes[1, 0]
    ax.imshow(feats["mfcc"], aspect="auto", origin="lower", cmap="magma")
    ax.set_title("MFCC (20)"); ax.set_xlabel("Frames"); ax.set_ylabel("Coeff")

    # Chroma
    ax = axes[1, 1]
    ax.imshow(feats["chroma"], aspect="auto", origin="lower", cmap="magma")
    ax.set_title("Chroma"); ax.set_xlabel("Frames"); ax.set_ylabel("Pitch class")

    # Spectral Contrast
    ax = axes[2, 0]
    ax.imshow(feats["contrast"], aspect="auto", origin="lower", cmap="magma")
    ax.set_title("Spectral Contrast"); ax.set_xlabel("Frames"); ax.set_ylabel("Bands")

    # Temporal (RMS & ZCR)
    ax = axes[2, 1]
    ax.plot(frames, feats["zcr"], label="ZCR")
    ax.plot(frames, feats["rms"], label="RMS")
    ax.set_title("Temporal Features"); ax.set_xlabel("Frames"); ax.set_ylabel("Value")
    ax.legend(frameon=False)

    out_png = Path(out_png)
    ensure_dir(out_png.parent)
    fig.savefig(out_png, dpi=350)
    plt.close(fig)
    print(f"Saved: {out_png}")

def pick_example(df, label):
    subset = df[df["label"] == label]
    if len(subset) == 0:
        # fallback: any sample
        return df.iloc[0]["path"], None
    return subset.iloc[0]["path"], label

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", required=True, help="folder for figures")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--neg_label", type=int, default=0)
    ap.add_argument("--pos_label", type=int, default=1)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if not {"path", "label"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: path,label")

    # One negative (healthy) and one positive (COPD)
    neg_path, _ = pick_example(df, args.neg_label)
    pos_path, _ = pick_example(df, args.pos_label)

    for path, tag in [(neg_path, "negative"), (pos_path, "positive")]:
        y, sr = load_audio(path, args.sr, args.duration)
        feats = compute_features(y, sr)
        title = Path(path).name
        out_png = Path(args.outdir) / f"features_{tag}.png"
        plot_gallery(y, sr, feats, title, out_png)

if __name__ == "__main__":
    main()
