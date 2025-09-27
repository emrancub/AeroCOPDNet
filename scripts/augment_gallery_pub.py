# scripts/augment_gallery_pub.py
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # prevent Windows OpenMP crash

import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------- Style ----------
mpl.rcParams.update({
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "figure.dpi": 350
})

# ---------- I/O ----------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _resample_science(y, sr0, sr):
    """Prefer scipy.polyphase; fallback to linear if scipy not present."""
    if sr0 == sr:
        return y
    try:
        from scipy.signal import resample_poly
        g = math.gcd(int(sr0), int(sr))
        up = int(sr // g)
        down = int(sr0 // g)
        return resample_poly(y, up, down)
    except Exception:
        # Linear fallback (good enough for visualization)
        n_new = int(round(len(y) * float(sr) / float(sr0)))
        x_old = np.linspace(0.0, 1.0, num=len(y), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_new, endpoint=False)
        return np.interp(x_new, x_old, y).astype(y.dtype, copy=False)

def load_audio_fixed(path: str, sr: int, duration_s: float, seed: int = 42):
    """Load mono, resample w/out resampy, fix length, normalize to [-1,1]."""
    y, sr0 = librosa.load(path, sr=None, mono=True)
    if sr0 != sr:
        y = _resample_science(y, sr0, sr)
    target = int(sr * duration_s)
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    else:
        y = y[:target]
    m = np.max(np.abs(y)) + 1e-8
    return (y / m), sr

# ---------- Features ----------
def mel_db(y, sr, n_fft=1024, hop=256, n_mels=128):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                       hop_length=hop, n_mels=n_mels, power=2.0)
    Sdb = librosa.power_to_db(S, ref=np.max)
    return Sdb, hop

# ---------- Wave-level aug ----------
def rand_time_shift(y, max_shift_ratio=0.12, rng=None):
    if rng is None: rng = np.random.default_rng()
    n = len(y); max_shift = int(max_shift_ratio * n)
    shift = int(rng.integers(-max_shift, max_shift + 1))
    if shift == 0: return y
    if shift > 0:  return np.concatenate([np.zeros(shift, dtype=y.dtype), y[:-shift]])
    else:          return np.concatenate([y[-shift:], np.zeros(-shift, dtype=y.dtype)])

def add_gaussian_noise(y, snr_db=18.0, rng=None):
    if rng is None: rng = np.random.default_rng()
    p_signal = np.mean(y**2) + 1e-12
    p_noise = p_signal / (10.0**(snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(p_noise), size=y.shape)
    out = y + noise
    return np.clip(out, -1.0, 1.0)

def rand_gain_db(y, gain_db_range=(-4, 4), rng=None):
    if rng is None: rng = np.random.default_rng()
    g = rng.uniform(*gain_db_range)
    out = y * (10.0**(g/20.0))
    m = np.max(np.abs(out)) + 1e-8
    return out / m

def mild_time_stretch(y, rate_range=(0.97, 1.03), target_len=None, rng=None):
    if rng is None: rng = np.random.default_rng()
    r = rng.uniform(*rate_range)
    y2 = librosa.effects.time_stretch(y, rate=r)
    if target_len is None:
        target_len = len(y)
    if len(y2) < target_len:
        y2 = np.pad(y2, (0, target_len - len(y2)))
    else:
        y2 = y2[:target_len]
    m = np.max(np.abs(y2)) + 1e-8
    return y2 / m

def wave_augment_pipeline(y, sr, rng=None):
    if rng is None: rng = np.random.default_rng()
    y1 = rand_time_shift(y, rng=rng)
    y1 = mild_time_stretch(y1, target_len=len(y), rng=rng)
    y1 = add_gaussian_noise(y1, rng=rng)
    y1 = rand_gain_db(y1, rng=rng)
    return y1

# ---------- SpecAug on mel-dB ----------
def freq_mask(mel_db_mat, F=18, n_masks=2, rng=None):
    if rng is None: rng = np.random.default_rng()
    S = mel_db_mat.copy(); v, t = S.shape; vmin = S.min()
    for _ in range(n_masks):
        f = rng.integers(0, F+1); f0 = rng.integers(0, max(1, v - f))
        S[f0:f0+f, :] = vmin
    return S

def time_mask(mel_db_mat, T=35, n_masks=2, rng=None):
    if rng is None: rng = np.random.default_rng()
    S = mel_db_mat.copy(); v, t = S.shape; vmin = S.min()
    for _ in range(n_masks):
        tt = rng.integers(0, T+1); t0 = rng.integers(0, max(1, t - tt))
        S[:, t0:t0+tt] = vmin
    return S

# ---------- Mixup ----------
def mixup(y1, y2, alpha=0.2, rng=None):
    if rng is None: rng = np.random.default_rng()
    lam = float(np.random.default_rng().beta(alpha, alpha))
    y = lam * y1 + (1.0 - lam) * y2
    m = np.max(np.abs(y)) + 1e-8
    return (y / m), lam

# ---------- Plotting ----------
def plot_aug_gallery(y_orig, y_wave, mel_orig, mel_fmask, mel_tmask,
                     y_mix, mel_mix, sr, title_prefix, out_png):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6), constrained_layout=True)

    # Row 1: waveforms
    t = np.arange(len(y_orig)) / sr
    axes[0,0].plot(t, y_orig, linewidth=0.9); axes[0,0].set_title("Original (wave)")
    axes[0,0].set_xlabel("Time (s)"); axes[0,0].set_ylabel("Amp")

    axes[0,1].plot(t, y_wave, linewidth=0.9); axes[0,1].set_title("WaveAug (wave)")
    axes[0,1].set_xlabel("Time (s)")

    axes[0,2].axis("off")

    axes[0,3].plot(t, y_mix, linewidth=0.9); axes[0,3].set_title("Mixup (wave)")
    axes[0,3].set_xlabel("Time (s)")

    axes[0,4].axis("off")

    # Row 2: spectrograms
    for ax, S, ttl in [
        (axes[1,0], mel_orig, "Original (log-mel)"),
        (axes[1,1], mel_db(y_wave, sr)[0], "WaveAug (log-mel)"),
        (axes[1,2], mel_fmask, "SpecAug: FreqMask"),
        (axes[1,3], mel_tmask, "SpecAug: TimeMask"),
        (axes[1,4], mel_mix,  "Mixup (log-mel)"),
    ]:
        im = ax.imshow(S, aspect="auto", origin="lower", cmap="magma")
        ax.set_title(ttl); ax.set_xlabel("Frames")
    axes[1,0].set_ylabel("Mel bins")

    ensure_dir(Path(out_png).parent)
    fig.suptitle(title_prefix, y=1.02, fontsize=13)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")

# ---------- Data helpers ----------
def pick_by_label(df, label):
    sub = df[df["label"] == label]
    if len(sub) == 0:
        return None
    return sub.sample(n=1, random_state=17).iloc[0]["path"]

def pick_partner_audio(df, current_path, target_len, sr, seed=1234):
    rng = np.random.default_rng(seed)
    other = df[df["path"] != current_path]
    if len(other) == 0:
        other = df.copy()
    partner = other.sample(n=1, random_state=int(rng.integers(0, 10_000))).iloc[0]["path"]
    y2, _ = load_audio_fixed(partner, sr, target_len / sr)
    return y2, partner

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with columns path,label")
    ap.add_argument("--outdir", required=True, help="Output folder for figures")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--neg_label", type=int, default=0)
    ap.add_argument("--pos_label", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if not {"path", "label"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: path,label")

    rng = np.random.default_rng(args.seed)

    for label, tag in [(args.neg_label, "negative"), (args.pos_label, "positive")]:
        p = pick_by_label(df, label)
        if p is None:
            print(f"[WARN] No sample with label={label} in {args.csv}. Skipping.")
            continue

        y, sr = load_audio_fixed(p, args.sr, args.duration, seed=args.seed)
        mel0, _ = mel_db(y, sr)

        # WaveAug
        y_w = wave_augment_pipeline(y, sr, rng=rng)

        # SpecAug (on mel)
        mel_f = freq_mask(mel0, F=18, n_masks=2, rng=rng)
        mel_t = time_mask(mel0, T=35, n_masks=2, rng=rng)

        # Mixup
        y2, partner_path = pick_partner_audio(df, p, target_len=len(y), sr=sr, seed=args.seed + 11)
        y_m, lam = mixup(y, y2, alpha=0.2, rng=rng)
        mel_m, _ = mel_db(y_m, sr)

        title = f"{Path(p).name}  |  mixup λ={lam:.2f}"
        out_png = Path(args.outdir) / f"augment_gallery_{tag}.png"
        plot_aug_gallery(y, y_w, mel0, mel_f, mel_t, y_m, mel_m, sr, title, out_png)

if __name__ == "__main__":
    main()
