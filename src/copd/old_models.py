# src/copd/old_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------- Small CNN backbone for spectrograms --------
def conv_block(in_ch, out_ch, k=3, s=1, p=1, pool=True):
    layers = [
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(kernel_size=2))  # downsample F,T by 2
    return nn.Sequential(*layers)

class DefaultBackbone(nn.Module):
    """Simple, stable CNN feature extractor for (B,1,F,T)."""
    def __init__(self, in_ch: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            conv_block(in_ch,   32, pool=True),   # -> (32, F/2,  T/2)
            conv_block(32,      64, pool=True),   # -> (64, F/4,  T/4)
            conv_block(64,     128, pool=True),   # -> (128,F/8,  T/8)
        )

    def forward(self, x):         # x: (B,1,F,T)
        return self.net(x)        # -> (B,128,F',T')

    @property
    def out_channels(self) -> int:
        # last block out_ch
        return 128

# -------- AeroCOPDNet Lite with spatial attention --------
class AeroCOPDNetLite(nn.Module):
    """
    Backbone (B,Cb,F',T') -> 1x1 proj to 256 -> 2D attention -> GAP -> MLP -> logit
    Works for binary (n_classes=1) or multi-class (n_classes>1).
    """
    def __init__(self, in_ch: int = 1, n_classes: int = 1, dropout: float = 0.2, attn_channels: int = 256):
        super().__init__()
        # build default backbone
        self.backbone = DefaultBackbone(in_ch)
        cb = self.backbone.out_channels

        # project to fixed channels for attention
        if cb == attn_channels:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Conv2d(cb, attn_channels, kernel_size=1, bias=True)

        # attention over F'×T'
        self.attn = nn.Sequential(
            nn.Conv2d(attn_channels, 64, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        # classifier head
        self.fc = nn.Sequential(
            nn.Linear(attn_channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, n_classes)  # 1 for binary logit
        )
        self.n_classes = n_classes

    def forward(self, x):
        # x: (B,1,F,T)
        z = self.backbone(x)      # (B, Cb, F', T')
        z = self.proj(z)          # (B, 256, F', T')

        a = self.attn(z)          # (B, 1, F', T')
        z = z * a                 # attention-weighted features

        z = z.mean(dim=(2, 3))    # GAP -> (B, 256)
        logit = self.fc(z).squeeze(-1) if self.n_classes == 1 else self.fc(z)
        return logit

# -------- factory --------
def build_model(name: str, in_ch: int = 1, n_classes: int = 1, dropout: float = 0.2):
    name = name.lower()
    if name == "aerocpdnet":
        return AeroCOPDNetLite(in_ch=in_ch, n_classes=n_classes, dropout=dropout)
    raise ValueError(f"Unknown model: {name}")
