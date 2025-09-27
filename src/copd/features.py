# src/copd/features.py
import torch
import torch.nn as nn
import torchaudio

# ---------- Utilities ----------

def _ensure_wave_3d(wav):
    """
    Accepts (L,), (1,L), (B,L), (B,1,L) and returns (B,1,L) on CPU float32.
    """
    if wav.ndim == 1:            # (L,)
        wav = wav.unsqueeze(0).unsqueeze(0)
    elif wav.ndim == 2:          # (B,L) or (1,L)
        wav = wav.unsqueeze(1)   # -> (B,1,L)
    elif wav.ndim == 3:          # (B,1,L)
        pass
    else:
        raise ValueError(f"Unexpected waveform shape: {tuple(wav.shape)}")
    return wav.detach().to("cpu", dtype=torch.float32)

def _add_channel_dim(spec):
    """
    torchaudio returns (B,F,T). Add channel to get (B,1,F,T).
    """
    if spec.ndim != 3:
        raise ValueError(f"Expected 3D spec (B,F,T), got {tuple(spec.shape)}")
    return spec.unsqueeze(1)

# ---------- Feature extractors ----------

class MelFeature(nn.Module):
    """
    Log-Mel spectrogram -> (B,1,n_mels,T) on CPU.
    """
    def __init__(self, sample_rate=16000, n_mels=128, n_fft=1024, hop_length=256, win_length=None,
                 f_min=0.0, f_max=None, power=2.0):
        super().__init__()
        self.sample_rate = sample_rate
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
            win_length=win_length, f_min=f_min, f_max=f_max, n_mels=n_mels, power=power
        )
        self.amptodb = torchaudio.transforms.AmplitudeToDB(stype="power")  # log compression

    def forward(self, wav):
        wav = _ensure_wave_3d(wav)              # (B,1,L) cpu
        B, C, L = wav.shape
        wav = wav.reshape(B*C, L)               # (B, L)
        spec = self.melspec(wav)                # (B, n_mels, T)
        spec = self.amptodb(spec)               # log-mel
        spec = _add_channel_dim(spec)           # (B,1,F,T)
        return spec


class MFCCFeature(nn.Module):
    """
    MFCC -> (B,1,n_mfcc,T) on CPU.
    """
    def __init__(self, sample_rate=16000, n_mfcc=20, melkwargs=None):
        super().__init__()
        if melkwargs is None:
            melkwargs = dict(n_fft=1024, hop_length=256, n_mels=128)
        self.sample_rate = sample_rate
        self.mfcc = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc, melkwargs=melkwargs)

    def forward(self, wav):
        wav = _ensure_wave_3d(wav)              # (B,1,L) cpu
        B, C, L = wav.shape
        wav = wav.reshape(B*C, L)               # (B, L)
        mfcc = self.mfcc(wav)                   # (B, n_mfcc, T)
        mfcc = _add_channel_dim(mfcc)           # (B,1,F,T)
        return mfcc


# ---------- Factory ----------

def build_feature(name, **kwargs):
    """
    name: 'mel' or 'mfcc'
    kwargs may include sample_rate, etc. (ignored if not needed)
    """
    name = str(name).lower()
    if name == "mel":
        return MelFeature(**kwargs)
    if name == "mfcc":
        return MFCCFeature(**kwargs)
    raise ValueError(f"Unknown feature type: {name}")
