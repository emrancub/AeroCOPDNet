import os, csv
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import librosa

class AudioBinaryDataset(Dataset):
    """
    Reads CSV with columns: path,label[,patient_id]
    Loads audio -> preprocess -> feature extraction done outside (features.py) OR here as raw waveform.
    We'll return (waveform, label, meta) and let collate / feature module convert to spectrogram.
    """
    def __init__(self, csv_path, sample_rate=16000, duration=4.0, mono=True,
                 return_path=False, patient_wise_id=False):
        self.df = pd.read_csv(csv_path)
        # flexible column names
        cols = {c.lower(): c for c in self.df.columns}
        path_col = cols.get("path") or cols.get("filepath") or cols.get("file")
        label_col = cols.get("label")
        self.pid_col = cols.get("patient_id") or cols.get("pid") or None
        if not path_col or not label_col:
            raise RuntimeError(f"CSV must have path & label columns. Got {list(self.df.columns)}")
        self.path_col = path_col; self.label_col = label_col
        self.return_path = return_path
        self.sr = sample_rate; self.samples = int(sample_rate*duration)
        self.mono = mono
        self.df = self.df[[self.path_col, self.label_col] + ([self.pid_col] if self.pid_col else [])].copy()
        self.df[self.label_col] = self.df[self.label_col].astype(int)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = str(row[self.path_col])
        label = int(row[self.label_col])
        # load audio
        y, sr = librosa.load(path, sr=self.sr, mono=self.mono)
        # trim/pad to fixed length
        if len(y) < self.samples:
            y = np.pad(y, (0, self.samples - len(y)))
        else:
            y = y[:self.samples]
        y = y.astype(np.float32)
        if self.return_path:
            return y, label, path
        return y, label

def build_sampler(labels):
    # Weighted sampler for imbalance
    class_count = np.bincount(labels, minlength=2)
    weights_per_class = 1.0 / np.maximum(1, class_count)
    weights = weights_per_class[labels]
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)

def make_loaders(csv_path, batch_size=32, num_workers=0, sample_rate=16000, duration=4.0,
                 weighted_sampling=True, return_path=False):
    ds = AudioBinaryDataset(csv_path, sample_rate=sample_rate, duration=duration, return_path=return_path)
    labels = ds.df[ds.label_col].values
    if weighted_sampling:
        sampler = build_sampler(labels)
        train_loader = DataLoader(ds, batch_size=batch_size, sampler=sampler,
                                  num_workers=num_workers, pin_memory=True, drop_last=False)
    else:
        train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True, drop_last=False)
    # For simplicity, we will split train/val inside train script; here we return the dataset object.
    return ds
