# src/copd/models.py

import torch
import torch.nn as nn
import timm
import math
import torch.nn.functional as F

# If you already have AeroCOPDNetLite and SmallBackbone classes, keep them.
# I will not repeat them here to avoid duplication.
# --------------------------------------------------
# ------------------------------
# Utils
# ------------------------------
def _conv_bn_act(in_ch, out_ch, k=3, s=(1,1), p=1, groups=1, act=True):
    m = [
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, groups=groups, bias=False),
        nn.BatchNorm2d(out_ch),
    ]
    if act:
        m.append(nn.SiLU(inplace=True))
    return nn.Sequential(*m)

class SqueezeExcite(nn.Module):
    def __init__(self, ch, rd=8):
        super().__init__()
        rd_ch = max(4, ch // rd)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(ch, rd_ch, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(rd_ch, ch, 1, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(self.avg(x))
        return x * w

class DSConvSE(nn.Module):
    """Depthwise-Separable Conv + SE with optional residual."""
    def __init__(self, in_ch, out_ch, stride=(1,1), se=True, drop_path=0.0):
        super().__init__()
        self.use_res = (in_ch == out_ch) and (stride == (1,1))
        self.dw = _conv_bn_act(in_ch, in_ch, k=3, s=stride, p=1, groups=in_ch, act=True)
        self.pw = _conv_bn_act(in_ch, out_ch, k=1, s=(1,1), p=0, groups=1, act=True)
        self.se = SqueezeExcite(out_ch) if se else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
    def forward(self, x):
        out = self.pw(self.dw(x))
        out = self.se(out)
        out = self.drop_path(out)
        if self.use_res:
            out = out + x
        return out

class DropPath(nn.Module):
    """Stochastic depth (per-sample) — safe, tiny regularizer."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand = x.new_empty(shape).bernoulli_(keep) / keep
        return x * rand

# ------------------------------
# Positional encoding (sine–cosine) for Transformer
# ------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (Tmax, C)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / max_len))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(1)  # (Tmax, 1, C)
        self.register_buffer('pe', pe, persistent=False)
    def forward(self, x):  # x: (T, B, C)
        T = x.size(0)
        return x + self.pe[:T]

# ------------------------------
# AeroCOPDNetLite (proposed)
# Input:  (B, 1, F, T)  e.g., F=128 mel bins, T variable
# Output: (B, 1)  logit
# ------------------------------
class AeroCOPDNetLite(nn.Module):
    def __init__(
        self,
        in_ch: int = 1,
        n_classes: int = 1,
        dropout: float = 0.3,
        width: int = 32,
        drop_path: float = 0.05,
        trans_dim: int = 192,
        trans_heads: int = 4,
        trans_layers: int = 2,
        trans_ff_mult: int = 4,
        time_pool: str = "meanmax",     # "mean" | "meanmax"
    ):
        super().__init__()
        assert n_classes == 1, "AeroCOPDNetLite returns a single logit."

        # --- CNN stem (downsample time a bit, keep freq)
        self.stem = nn.Sequential(
            _conv_bn_act(in_ch, width, k=5, s=(1,2), p=2),   # stride time=2, freq=1
            DSConvSE(width, width, stride=(1,1), se=True, drop_path=0.0),
        )

        # --- Stages: reduce frequency, keep time
        ch1, ch2, ch3 = width, width*2, width*4
        # reduce F by 2 (stride on freq axis), keep T
        self.stage1 = nn.Sequential(
            DSConvSE(ch1, ch2, stride=(2,1), se=True, drop_path=drop_path*0.3),
            DSConvSE(ch2, ch2, stride=(1,1), se=True, drop_path=drop_path*0.3),
        )
        # reduce F again
        self.stage2 = nn.Sequential(
            DSConvSE(ch2, ch3, stride=(2,1), se=True, drop_path=drop_path*0.6),
            DSConvSE(ch3, ch3, stride=(1,1), se=True, drop_path=drop_path*0.6),
        )

        # --- squeeze frequency -> (B, C, 1, T) -> (B, T, C)
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))  # keep time dimension

        # --- linear proj into transformer dim (if needed)
        self.proj = nn.Linear(ch3, trans_dim)

        # --- Transformer encoder over time
        enc_layer = nn.TransformerEncoderLayer(
            d_model=trans_dim,
            nhead=trans_heads,
            dim_feedforward=trans_ff_mult * trans_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=False,   # we feed (T,B,C)
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer=enc_layer, num_layers=trans_layers)
        self.pos_enc = PositionalEncoding(trans_dim, max_len=4000)

        # --- temporal pooling & head
        self.time_pool = time_pool
        head_in = trans_dim if time_pool == "mean" else (2 * trans_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Dropout(dropout),
            nn.Linear(head_in, n_classes)
        )

    def forward(self, x):         # x: (B,1,F,T)
        # CNN
        x = self.stem(x)          # (B, C1, F, T/2)
        x = self.stage1(x)        # (B, C2, F/2, T/2)
        x = self.stage2(x)        # (B, C3, F/4, T/2)

        # squeeze frequency, shape -> (B, C3, 1, T')
        x = self.freq_pool(x)     # (B, C3, 1, T')
        x = x.squeeze(2)          # (B, C3, T')
        x = x.transpose(1, 2)     # (B, T', C3)

        # project to transformer dim
        x = self.proj(x)          # (B, T', D)

        # transformer expects (T, B, D)
        x = x.transpose(0, 1)     # (T', B, D)
        x = self.pos_enc(x)
        x = self.transformer(x)   # (T', B, D)
        x = x.transpose(0, 1)     # (B, T', D)

        # temporal pooling
        if self.time_pool == "mean":
            h = x.mean(dim=1)     # (B, D)
        else:
            h_mean = x.mean(dim=1)
            h_max  = x.max(dim=1).values
            h = torch.cat([h_mean, h_max], dim=1)  # (B, 2D)

        # head -> logit
        logits = self.head(h)     # (B, 1)
        return logits

# ==== AeroCOPDNetLite v2 (multiscale CNN + SE + ASP) ====
import torch, torch.nn as nn, torch.nn.functional as F

# --- tiny building blocks ---
class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, d=1, groups=1, act="gelu"):
        super().__init__()
        p = ((k - 1) // 2) * d
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p,
                              dilation=d, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU() if act == "gelu" else nn.ReLU(inplace=True)
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class SE(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        m = max(8, c // r)
        self.fc1 = nn.Conv2d(c, m, 1, bias=True)
        self.fc2 = nn.Conv2d(m, c, 1, bias=True)
    def forward(self, x):
        s = x.mean(dim=(2,3), keepdim=True)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s

class ResidualSE(nn.Module):
    def __init__(self, c_in, c_out, stride=1, dilation=1, drop=0.0):
        super().__init__()
        self.conv1 = ConvBNAct(c_in,  c_out, k=3, s=stride, d=dilation)
        self.conv2 = ConvBNAct(c_out, c_out, k=3, s=1,      d=dilation)
        self.se = SE(c_out)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

        # --- IMPORTANT: use 1x1 conv with stride to match main path shapes exactly ---
        if stride > 1 or c_in != c_out:
            self.down = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(c_out)
            )
        else:
            self.down = nn.Identity()

    def _align(self, a, b):
        # Optional safety: center-crop the larger one if a tiny off-by-one remains
        if a.shape[-2:] == b.shape[-2:]:
            return a, b
        Fa, Ta = a.shape[-2], a.shape[-1]
        Fb, Tb = b.shape[-2], b.shape[-1]
        Fm, Tm = min(Fa, Fb), min(Ta, Tb)

        def center_crop(x, Fm, Tm):
            F0 = (x.shape[-2] - Fm) // 2
            T0 = (x.shape[-1] - Tm) // 2
            return x[..., F0:F0+Fm, T0:T0+Tm]

        return center_crop(a, Fm, Tm), center_crop(b, Fm, Tm)

    def forward(self, x):
        idn = self.down(x)
        z = self.conv1(x)
        z = self.conv2(z)
        z = self.se(z)
        z = self.drop(z)
        # Safety align if shapes differ by 1 due to odd sizes
        idn, z = self._align(idn, z)
        return F.gelu(z + idn)


class MultiScaleStem(nn.Module):
    """Parallel 3×3, 5×5, dilated 3×3, and 1×1; then fuse + SE."""
    def __init__(self, in_ch=1, out_ch=96):
        super().__init__()
        b = out_ch // 4
        self.b1 = ConvBNAct(in_ch, b, k=3)
        self.b2 = ConvBNAct(in_ch, b, k=5)
        self.b3 = ConvBNAct(in_ch, b, k=3, d=2)
        self.b4 = ConvBNAct(in_ch, out_ch - 3*b, k=1)
        self.se = SE(out_ch)
    def forward(self, x):
        z = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)
        return self.se(z)

class AttentiveStatsPool(nn.Module):
    """x: (B,C,T) -> concat(mean,std) with learned attention over T."""
    def __init__(self, in_dim, bottleneck=128):
        super().__init__()
        self.att = nn.Sequential(
            nn.Conv1d(in_dim, bottleneck, 1), nn.Tanh(),
            nn.Conv1d(bottleneck, 1, 1)
        )
    def forward(self, x):
        # x: (B,C,T)
        w = self.att(x)                    # (B,1,T)
        w = torch.softmax(w, dim=-1)
        mu = torch.sum(x * w, dim=-1)      # (B,C)
        s2 = torch.sum(x * x * w, dim=-1) - mu * mu
        std = torch.sqrt(torch.clamp(s2, min=1e-6))
        return torch.cat([mu, std], dim=1) # (B,2C)

def _delta_channels(x):
    """Make Δ and Δ² along time; x: (B,1,F,T). Returns (B,3,F,T)."""
    # pad 1 frame on both ends, central differences
    xp = F.pad(x, (1,1,0,0), mode="replicate")
    d1 = 0.5 * (xp[..., 2:] - xp[..., :-2])
    xp2 = F.pad(d1, (1,1,0,0), mode="replicate")
    d2 = xp2[..., 2:] - xp2[..., :-2]
    return torch.cat([x, d1, d2], dim=1)

class AeroCOPDNetLiteV2(nn.Module):
    """
    Input: (B,1,F=128,T~250)
    Stem: multi-scale CNN -> Residual SE stages (downsample in time/freq)
    Pool: AdaptiveAvg over F -> (B,C,T); AttentiveStatsPool over T -> (B,2C)
    Head: Dropout + Linear -> 1 logit
    """
    def __init__(self, in_ch=1, n_classes=1, dropout=0.25,
                 base=96, use_deltas=True, asp_bottleneck=128):
        super().__init__()
        assert n_classes == 1, "Binary head (1 logit)."
        self.use_deltas = use_deltas

        stem_in = in_ch if not use_deltas else 3
        self.delta = _delta_channels if use_deltas else (lambda t: t)
        self.stem  = MultiScaleStem(stem_in, out_ch=base)                # (B,base,F,T)

        # three stages with gentle downsampling (x2, x2, x2) ~ keep time fairly long
        self.stage1 = nn.Sequential(
            ResidualSE(base,   base,   stride=1, drop=dropout*0.25),
            ResidualSE(base,   base,   stride=1, drop=0.0),
        )
        self.down1  = ResidualSE(base,  base*2, stride=2, drop=dropout*0.25)  # (F/2, T/2)
        self.stage2 = nn.Sequential(
            ResidualSE(base*2, base*2, stride=1, drop=dropout*0.25),
            ResidualSE(base*2, base*2, stride=1, drop=0.0),
        )
        self.down2  = ResidualSE(base*2, base*3, stride=2, drop=dropout*0.25) # (F/4, T/4)
        self.stage3 = nn.Sequential(
            ResidualSE(base*3, base*3, stride=1, drop=dropout*0.25),
            ResidualSE(base*3, base*3, stride=1, drop=0.0),
        )

        # squeeze frequency -> sequence over time
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))   # (B,C,1,T)
        self.proj = nn.Conv1d(base*3, base*3, kernel_size=1, bias=False)

        self.asp = AttentiveStatsPool(base*3, bottleneck=asp_bottleneck) # (B,2C)
        self.head = nn.Sequential(
            nn.LayerNorm(base*3*2),
            nn.Dropout(dropout),
            nn.Linear(base*3*2, n_classes)
        )

    def forward(self, x):                     # x: (B,1,F,T)
        x = self.delta(x)                     # -> (B,3,F,T) if use_deltas
        z = self.stem(x)                      # (B,C,F,T)
        z = self.stage1(z); z = self.down1(z)
        z = self.stage2(z); z = self.down2(z)
        z = self.stage3(z)                    # (B,C',F',T')

        z = self.freq_pool(z).squeeze(2)      # (B,C',T')
        z = self.proj(z)                      # (B,C',T')
        z = self.asp(z)                       # (B,2C')
        logits = self.head(z)                 # (B,1)
        return logits

# ------------------------------
# ResNet18Audio baseline, MobileNet-style + Attentive Stats Pooling network
# ------------------------------
# --- Lightweight AeroCOPDNetLite-Nano: MobileNet-style + Attentive Stats Pooling ---
import torch
import torch.nn as nn
import torch.nn.functional as F

class _ConvBNAct(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=(1,1), p=None, groups=1, act="silu"):
        super().__init__()
        if p is None:
            p = (k // 2, k // 2) if isinstance(k, int) else (k[0] // 2, k[1] // 2)
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True) if act == "silu" else nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class _SE(nn.Module):
    """Squeeze-Excitation with tiny bottleneck."""
    def __init__(self, c, r=0.25):
        super().__init__()
        h = max(8, int(c * r))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, h, 1, bias=True), nn.SiLU(inplace=True),
            nn.Conv2d(h, c, 1, bias=True), nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(self.pool(x))
        return x * w

class _MBConv(nn.Module):
    """
    Inverted residual block (MobileNetV2/MBConv) with depthwise + SE and safe skip.
    Stride may be a tuple, e.g., (1,2) to downsample time only.
    """
    def __init__(self, c_in, c_out, stride=(1,1), expand=4, se_ratio=0.25, drop=0.0):
        super().__init__()
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        hidden = int(c_in * expand)

        self.use_res = (self.stride == (1,1) and c_in == c_out)
        self.expand = (expand != 1)

        # 1x1 expand
        self.pw1 = _ConvBNAct(c_in, hidden, k=1, s=(1,1)) if self.expand else nn.Identity()
        # 3x3 depthwise
        self.dw  = _ConvBNAct(hidden if self.expand else c_in, hidden, k=3, s=self.stride, groups=(hidden if self.expand else c_in))
        # SE
        self.se  = _SE(hidden)
        # 1x1 project (linear)
        self.pw2 = nn.Sequential(
            nn.Conv2d(hidden, c_out, 1, bias=False),
            nn.BatchNorm2d(c_out),
        )
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

        # downsample/align for skip when shapes differ
        if not self.use_res:
            self.skip = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=1, stride=self.stride, bias=False),
                nn.BatchNorm2d(c_out),
            )
        else:
            self.skip = nn.Identity()

    def _align(self, a, b):
        # center-crop larger spatial dims to match (handles odd lengths safely)
        if a.shape[-2:] == b.shape[-2:]:
            return a, b
        Fa, Ta = a.shape[-2], a.shape[-1]
        Fb, Tb = b.shape[-2], b.shape[-1]
        Fm, Tm = min(Fa, Fb), min(Ta, Tb)
        def ccrop(x, Fm, Tm):
            f0 = (x.shape[-2] - Fm) // 2
            t0 = (x.shape[-1] - Tm) // 2
            return x[..., f0:f0+Fm, t0:t0+Tm]
        return ccrop(a, Fm, Tm), ccrop(b, Fm, Tm)

    def forward(self, x):
        idn = self.skip(x)
        z = x
        if self.expand:
            z = self.pw1(z)
        z = self.dw(z)
        z = self.se(z)
        z = self.pw2(z)
        z = self.drop(z)
        idn, z = self._align(idn, z)
        return F.silu(z + idn)  # residual

class _AttnStatsPool1D(nn.Module):
    """
    Attentive statistics pooling along time.
    Expects input (B, C, T) and returns (B, 2C) = [weighted-mean, weighted-std].
    """
    def __init__(self, c, attn_hidden=64, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.tok = nn.Sequential(
            nn.Conv1d(c, attn_hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(attn_hidden, 1, kernel_size=1)
        )
    def forward(self, x):           # x: (B,C,T)
        scores = self.tok(x)        # (B,1,T)
        w = torch.softmax(scores, dim=-1)
        mu = torch.sum(w * x, dim=-1)                              # (B,C)
        m2 = torch.sum(w * (x ** 2), dim=-1)                       # (B,C)
        var = (m2 - mu ** 2).clamp_min(self.eps)
        std = torch.sqrt(var + self.eps)                           # (B,C)
        return torch.cat([mu, std], dim=1)                         # (B,2C)

class AeroCOPDNetLiteNano(nn.Module):
    """
    Lightweight spectrogram classifier for (B,1,F,T):
      - Stem
      - MBConv stages with mostly time-only downsampling
      - Frequency squeeze, Attentive Stats Pool over T
      - Linear head to 1 logit
    """
    def __init__(self, in_ch=1, n_classes=1, dropout=0.2, base=48, expand=4, se_ratio=0.25, attn_hidden=64):
        super().__init__()
        assert n_classes == 1, "binary logit expected (n_classes=1)."

        # Stem: mild downsample both dims
        self.stem = _ConvBNAct(in_ch, base, k=5, s=(2,2))

        # Stages (choose strides; mostly downsample time to keep freq detail)
        self.stage1 = nn.Sequential(
            _MBConv(base, base, stride=(1,2), expand=expand, se_ratio=se_ratio, drop=dropout),
            _MBConv(base, base, stride=(1,1), expand=expand, se_ratio=se_ratio, drop=dropout),
        )
        self.stage2 = nn.Sequential(
            _MBConv(base, base*2, stride=(2,2), expand=expand, se_ratio=se_ratio, drop=dropout),
            _MBConv(base*2, base*2, stride=(1,1), expand=expand, se_ratio=se_ratio, drop=dropout),
        )
        self.stage3 = nn.Sequential(
            _MBConv(base*2, base*3, stride=(1,2), expand=expand, se_ratio=se_ratio, drop=dropout),
            _MBConv(base*3, base*3, stride=(1,1), expand=expand, se_ratio=se_ratio, drop=dropout),
        )
        self.stage4 = nn.Sequential(
            _MBConv(base*3, base*4, stride=(1,2), expand=expand, se_ratio=se_ratio, drop=dropout),
            _MBConv(base*4, base*4, stride=(1,1), expand=expand, se_ratio=se_ratio, drop=dropout),
        )

        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))  # (B,C,1,T)
        self.tpool = _AttnStatsPool1D(base*4, attn_hidden=attn_hidden)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(base*8, n_classes)  # 2*C -> 1
        )

    def forward(self, x):         # x: (B,1,F,T)
        z = self.stem(x)
        z = self.stage1(z)
        z = self.stage2(z)
        z = self.stage3(z)
        z = self.stage4(z)
        z = self.freq_pool(z).squeeze(2)   # (B,C,T)
        stats = self.tpool(z)              # (B,2C)
        logits = self.head(stats)          # (B,1)
        return logits

# ------------------------------
# Basic 2D CNN baseline (spectrogram in -> logits)
# ------------------------------
class BasicCNN2D(nn.Module):
    def __init__(self, in_ch=1, n_classes=1, dropout=0.3, base=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1), nn.BatchNorm2d(base), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(base, base*2, 3, padding=1), nn.BatchNorm2d(base*2), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(base*2, base*4, 3, padding=1), nn.BatchNorm2d(base*4), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base*4, n_classes)
        )
    def forward(self, x):           # x: (B,1,F,T)
        z = self.net(x)             # (B, C, 1, 1)
        logits = self.head(z)       # (B,1)
        return logits

# ------------------------------
# CRNN baseline: small CNN -> BiGRU -> linear
# ------------------------------
class CRNN(nn.Module):
    def __init__(self, in_ch=1, n_classes=1, rnn_hidden=128, rnn_layers=1, dropout=0.3, base=32, bidirectional=True):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1), nn.BatchNorm2d(base), nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(base, base*2, 3, padding=1), nn.BatchNorm2d(base*2), nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))  # squeeze F -> 1, keep T
        self.bi = 2 if bidirectional else 1
        self.rnn = nn.GRU(input_size=base*2, hidden_size=rnn_hidden,
                          num_layers=rnn_layers, batch_first=True,
                          dropout=0.0 if rnn_layers == 1 else dropout,
                          bidirectional=bidirectional)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.bi * rnn_hidden, n_classes)
        )

    def forward(self, x):                 # (B,1,F,T)
        z = self.cnn(x)                   # (B,C',F',T')
        z = self.freq_pool(z)             # (B,C',1,T')
        z = z.squeeze(2).transpose(1, 2)  # (B,T',C')
        _, h_n = self.rnn(z)              # h_n: (num_layers*bi, B, H)
        h_last = h_n[-1] if self.bi == 1 else torch.cat([h_n[-2], h_n[-1]], dim=1)  # (B, H) or (B, 2H)
        logits = self.head(h_last)        # (B,1)
        return logits

# ------------------------------
# LSTM / GRU “sequence over time” baseline
# We treat the spectrogram as a sequence of time steps each with F features.
# ------------------------------
class RNNOnSpec(nn.Module):
    def __init__(self, rnn_type="lstm", in_ch=1, n_classes=1, hidden=128, layers=1, dropout=0.3, bidirectional=True):
        super().__init__()
        assert rnn_type in ["lstm", "gru"]
        self.bi = 2 if bidirectional else 1
        self.rnn_type = rnn_type
        self.dropout = nn.Dropout(dropout)

        # project (C=1, F) into a feature dimension
        # we will average over C then feed (B,T,F) into RNN after a linear proj
        self.proj = nn.Linear(128, hidden)  # we'll reshape to guarantee feature=128 first with a conv
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=(5, 5), padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(5, 5), padding=2), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(5, 5), padding=2), nn.ReLU(),
        )
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=hidden, hidden_size=hidden, num_layers=layers,
                               batch_first=True, dropout=0.0 if layers == 1 else dropout,
                               bidirectional=bidirectional)
        else:
            self.rnn = nn.GRU(input_size=hidden, hidden_size=hidden, num_layers=layers,
                              batch_first=True, dropout=0.0 if layers == 1 else dropout,
                              bidirectional=bidirectional)
        self.fc = nn.Linear(self.bi * hidden, n_classes)

    def forward(self, x):                 # (B,1,F,T)
        z = self.conv(x)                  # (B,128,F,T)
        z = z.mean(dim=2)                 # avg over F -> (B,128,T)
        z = z.transpose(1, 2)             # (B,T,128)
        z = self.proj(z)                  # (B,T,H)
        out, h = self.rnn(z)
        h_last = h if self.rnn_type == "gru" else h[0]     # (num_layers*bi,B,H)
        h_last = h_last[-1] if self.bi == 1 else torch.cat([h_last[-2], h_last[-1]], dim=1)  # (B, bi*H)
        h_last = self.dropout(h_last)
        logits = self.fc(h_last)          # (B,1)
        return logits

# ------------------------------
# ResNet18 baseline (torchvision)
# Modifies the first conv to accept 1 channel; final fc -> 1 logit
# ------------------------------
class ResNet18Audio(nn.Module):
    def __init__(self, in_ch=1, n_classes=1, dropout=0.3):
        super().__init__()
        try:
            import torchvision.models as tvm
        except Exception as e:
            raise RuntimeError("torchvision is required for ResNet18Audio") from e

        self.backbone = tvm.resnet18(weights=None)
        # swap first conv to 1 channel
        self.backbone.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # replace fc with dropout + linear
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feats, n_classes)
        )

    def forward(self, x):     # (B,1,F,T) resized implicitly by resnet stages
        return self.backbone(x)

# ------------------------------
# Optional: Audio Spectrogram Transformer (AST) via timm
# If timm is not installed, raise a descriptive error.
# ------------------------------
class ASTAudio(nn.Module):
    def __init__(self, in_ch=1, n_classes=1, dropout=0.3, model_name="vit_base_patch16_224"):
        super().__init__()
        try:
            import timm
        except Exception as e:
            raise RuntimeError("timm is required for ASTAudio (pip install timm).") from e
        # Many ViT/AST variants expect 3 channels; we can expand 1->3
        self.expand = (in_ch == 1)
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=n_classes)
        if hasattr(self.backbone, "patch_embed") and getattr(self.backbone.patch_embed, "proj", None) is not None:
            # If the patch embed expects 3-ch, adapt by repeating the single channel
            pass

    def forward(self, x):   # (B,1,F,T)
        if self.expand:
            x = x.repeat(1, 3, 1, 1)  # (B,3,F,T)
        # Most ViTs expect square 224x224; your dataloader/feature step should resize if you really use AST.
        # Otherwise, keep this class for later extension.
        logits = self.backbone(x)
        return logits

# ------------------------------
# build_model factory
# ------------------------------
def build_model(name: str, in_ch: int = 1, n_classes: int = 1, dropout: float = 0.0, pretrained: bool = True):
    name = name.lower()
    # if name in ["aerocpdnet", "aerocpdnetlite"]:
    #     # your existing AeroCOPDNetLite (not shown here)
    #     return AeroCOPDNetLite(in_ch, n_classes=n_classes, dropout=dropout)
    # if name in ("aerocpdnetlite2", "aerocpdnetv2", "aerocpdnetlite_v2"):
    #     return AeroCOPDNetLiteV2(in_ch=in_ch, n_classes=n_classes, dropout=dropout)
    if name in ["aerocpdnet_nano", "aerocpdnetlite2", "aerocpdnetlite_nano", "aerocpdnetlitexs"]:
        return AeroCOPDNetLiteNano(in_ch=in_ch, n_classes=n_classes, dropout=dropout, base=48)
    if name == "basiccnn":
        return BasicCNN2D(in_ch, n_classes, dropout)
    if name == "crnn":
        return CRNN(in_ch, n_classes, dropout=dropout)
    if name == "lstm":
        return RNNOnSpec(rnn_type="lstm", in_ch=in_ch, n_classes=n_classes, dropout=dropout)
    if name == "gru":
        return RNNOnSpec(rnn_type="gru", in_ch=in_ch, n_classes=n_classes, dropout=dropout)
    if name == "resnet18":
        return ResNet18Audio(in_ch, n_classes, dropout)
    if name == "ast":
        return ASTAudio(in_ch, n_classes, dropout)

    raise ValueError(f"Unknown model name: {name}")
