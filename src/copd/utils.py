import os, random, math, time, json
from pathlib import Path
import numpy as np
import torch

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class AvgMeter:
    def __init__(self): self.reset()
    def reset(self): self.sum=0.0; self.count=0
    def update(self, v, n=1): self.sum += float(v)*n; self.count += n
    @property
    def avg(self): return self.sum / max(1,self.count)

def timestamp():
    return time.strftime("%Y%m%d_%H%M%S")

def save_json(d, path):
    with open(path, "w", encoding="utf-8") as f: json.dump(d, f, indent=2)

def device_str():
    return "cuda" if torch.cuda.is_available() else "cpu"
