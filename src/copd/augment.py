# src/copd/augment.py
import random
import torch
import torch.nn as nn
import torchaudio.functional as AF

# -------------- helpers --------------

def _to_3d_wav(x):
    # (L,) | (1,L) | (B,L) | (B,1,L) -> (B,1,L) on CPU float32
    if x.ndim == 1:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 2:
        x = x.unsqueeze(1)  # (B,1,L)
    elif x.ndim == 3:
        pass
    else:
        raise ValueError(f"Unexpected waveform shape: {tuple(x.shape)}")
    return x.detach().to("cpu", dtype=torch.float32)

def _to_4d_spec(x):
    # (F,T) | (1,F,T) | (B,1,F,T) -> (B,1,F,T)
    if x.ndim == 2:       # (F,T)
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 3:     # (1,F,T)
        x = x.unsqueeze(0)
    elif x.ndim == 4:     # already (B,1,F,T)
        pass
    else:
        raise ValueError(f"Unexpected spec shape: {tuple(x.shape)}")
    return x

def _same_shape_like(x, target_len):
    # center-crop or pad 1D wav to length target_len
    L = x.shape[-1]
    if L == target_len:
        return x
    if L > target_len:
        start = (L - target_len) // 2
        return x[..., start:start+target_len]
    # pad
    pad = target_len - L
    left = pad // 2
    right = pad - left
    return torch.nn.functional.pad(x, (left, right))


# -------------- Augmentations --------------

class WaveAugment(nn.Module):
    """
    Simple waveform-domain augmentation:
      - random gain (± dB)
      - add Gaussian noise to target SNR
      - random circular time shift
      - random time-stretch via resample
    Works on CPU. Input/Output shapes: (L,), (1,L), (B,1,L) -> same shape.
    """
    def __init__(self, sample_rate=16000,
                 gain_db=(-6, 6), noise_snr_db=(15, 30),
                 shift_ms=(-50, 50), stretch_factor=(0.90, 1.10),
                 p_gain=0.5, p_noise=0.5, p_shift=0.5, p_stretch=0.5):
        super().__init__()
        self.sr = sample_rate
        self.gain_db = gain_db
        self.noise_snr_db = noise_snr_db
        self.shift_ms = shift_ms
        self.stretch_factor = stretch_factor
        self.p_gain = p_gain
        self.p_noise = p_noise
        self.p_shift = p_shift
        self.p_stretch = p_stretch

    def forward(self, wav, sr=None):
        sr = sr or self.sr
        shape = wav.shape
        wav = _to_3d_wav(wav)  # (B,1,L)
        B, _, L = wav.shape

        # Gain
        if random.random() < self.p_gain:
            g = random.uniform(*self.gain_db)
            wav = wav * (10.0 ** (g / 20.0))

        # Add noise at SNR
        if random.random() < self.p_noise:
            snr = random.uniform(*self.noise_snr_db)
            rms = wav.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-8
            noise_rms = rms / (10.0 ** (snr / 20.0))
            noise = torch.randn_like(wav) * noise_rms
            wav = wav + noise

        # Time shift (circular)
        if random.random() < self.p_shift and self.shift_ms != (0, 0):
            max_shift = int(abs(self.shift_ms[1]) * sr / 1000.0)
            min_shift = -int(abs(self.shift_ms[0]) * sr / 1000.0)
            k = random.randint(min_shift, max_shift)
            wav = torch.roll(wav, shifts=k, dims=-1)

        # Time-stretch via resampling
        if random.random() < self.p_stretch:
            fac = random.uniform(*self.stretch_factor)
            new_len = max(1, int(L / fac))
            # resample each item independently (CPU)
            out = []
            for b in range(B):
                y = AF.resample(wav[b, 0], orig_freq=sr, new_freq=int(sr / fac))
                y = y.unsqueeze(0)  # (1, L')
                y = _same_shape_like(y, L)
                out.append(y)
            wav = torch.stack(out, dim=0)  # (B,1,L)

        return wav.reshape(shape)  # back to original shape


class SpecAugment(nn.Module):
    """
    Frequency & time masking. Accepts (F,T), (1,F,T), or (B,1,F,T) and returns same rank.
    """
    def __init__(self, freq_mask_param=24, time_mask_param=32, num_masks=2, p=1.0):
        super().__init__()
        self.F = freq_mask_param
        self.T = time_mask_param
        self.num_masks = num_masks
        self.p = p

    def forward(self, x):
        if random.random() >= self.p:
            return x
        orig_rank = x.ndim
        X = _to_4d_spec(x).clone()  # (B,1,F,T)
        B, C, F, T = X.shape

        for b in range(B):
            # frequency masks
            for _ in range(self.num_masks):
                f = random.randint(0, max(0, min(self.F, F-1)))
                if f > 0:
                    f0 = random.randint(0, F - f)
                    X[b, :, f0:f0+f, :] = 0.0
            # time masks
            for _ in range(self.num_masks):
                t = random.randint(0, max(0, min(self.T, T-1)))
                if t > 0:
                    t0 = random.randint(0, T - t)
                    X[b, :, :, t0:t0+t] = 0.0

        # restore original rank
        if orig_rank == 2:      # (F,T)
            return X[0, 0]
        if orig_rank == 3:      # (1,F,T)
            return X[0]
        return X                # (B,1,F,T)
