# src/copd/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# If you already define AeroCOPDNetLite elsewhere, keep it and its import.

# ------------------------------
# Basic 2D CNN baseline
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
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(base*4, n_classes))
    def forward(self, x):
        return self.head(self.net(x))

# ------------------------------
# CRNN baseline
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
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))
        self.bi = 2 if bidirectional else 1
        self.rnn = nn.GRU(input_size=base*2, hidden_size=rnn_hidden,
                          num_layers=rnn_layers, batch_first=True,
                          dropout=0.0 if rnn_layers == 1 else dropout,
                          bidirectional=bidirectional)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.bi * rnn_hidden, n_classes))

    def forward(self, x):
        z = self.cnn(x)                       # (B,C',F',T')
        z = self.freq_pool(z)                 # (B,C',1,T')
        z = z.squeeze(2).transpose(1, 2)      # (B,T',C')
        _, h_n = self.rnn(z)                  # (L*bi,B,H)
        h_last = h_n[-1] if self.bi == 1 else torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.head(h_last)

# ------------------------------
# RNN over spectrogram
# ------------------------------
class RNNOnSpec(nn.Module):
    def __init__(self, rnn_type="lstm", in_ch=1, n_classes=1, hidden=128, layers=1, dropout=0.3, bidirectional=True):
        super().__init__()
        assert rnn_type in ["lstm", "gru"]
        self.bi = 2 if bidirectional else 1
        self.rnn_type = rnn_type
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(128, hidden)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=(5,5), padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(5,5), padding=2), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(5,5), padding=2), nn.ReLU(),
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

    def forward(self, x):
        z = self.conv(x).mean(dim=2)          # (B,128,T)
        z = z.transpose(1, 2)                 # (B,T,128)
        z = self.proj(z)                      # (B,T,H)
        out, h = self.rnn(z)
        h_last = h if self.rnn_type == "gru" else h[0]
        h_last = h_last[-1] if self.bi == 1 else torch.cat([h_last[-2], h_last[-1]], dim=1)
        h_last = self.dropout(h_last)
        return self.fc(h_last)

# ------------------------------
# ResNet18 baseline
# ------------------------------
class ResNet18Audio(nn.Module):
    def __init__(self, in_ch=1, n_classes=1, dropout=0.3):
        super().__init__()
        import torchvision.models as tvm
        self.backbone = tvm.resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_feats, n_classes))
    def forward(self, x):
        return self.backbone(x)

# ------------------------------
# AST with fixed 128×256 grid (8×16 patches)
# ------------------------------
class ASTBinary(nn.Module):
    def __init__(self, in_ch=1, n_classes=1, dropout=0.0,
                 pretrained_backbone=True, target_freq_bins=128,
                 patch_size=16, target_time_patches=16):
        super().__init__()
        assert n_classes == 1, "ASTBinary is configured for single-logit binary classification."
        self.patch_size = int(patch_size)
        self.target_h = int(target_freq_bins)
        self.target_w = int(target_time_patches) * self.patch_size  # 16*16 = 256
        # Fix img_size -> pos_embed length stays constant (1 + 8*16)
        self.backbone = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained_backbone,
            in_chans=in_ch,
            img_size=(self.target_h, self.target_w),
            num_classes=n_classes,
            global_pool="avg",
            drop_rate=dropout,
        )

    @torch.no_grad()
    def _pad_or_crop(self, x):
        B, C, H, W = x.shape
        # F
        if H < self.target_h:
            x = F.pad(x, (0, 0, 0, self.target_h - H))
        elif H > self.target_h:
            top = (H - self.target_h) // 2
            x = x[:, :, top:top + self.target_h, :]
        # T
        if W < self.target_w:
            x = F.pad(x, (0, self.target_w - W, 0, 0))
        elif W > self.target_w:
            left = (W - self.target_w) // 2
            x = x[:, :, :, left:left + self.target_w]
        return x

    def forward(self, x):
        x = self._pad_or_crop(x)     # (B,1,128,256)
        return self.backbone(x)      # (B,1)

# ------------------------------
# build_model factory
# ------------------------------
def build_model(name: str, in_ch: int = 1, n_classes: int = 1, dropout: float = 0.3, pretrained: bool = True):
    name = name.lower()
    if name in ("basiccnn",):
        return BasicCNN2D(in_ch, n_classes, dropout)
    if name == "crnn":
        return CRNN(in_ch, n_classes, dropout=dropout)
    if name == "lstm":
        return RNNOnSpec(rnn_type="lstm", in_ch=in_ch, n_classes=n_classes, dropout=dropout)
    if name == "gru":
        return RNNOnSpec(rnn_type="gru", in_ch=in_ch, n_classes=n_classes, dropout=dropout)
    if name == "resnet18":
        return ResNet18Audio(in_ch, n_classes, dropout)
    if name in ("ast", "audio_spectrogram_transformer"):
        return ASTBinary(
            in_ch=in_ch, n_classes=n_classes, dropout=dropout,
            pretrained_backbone=pretrained, target_freq_bins=128,
            patch_size=16, target_time_patches=16
        )
    if name in ("aerocpdnet", "aerocpdnetlite"):
        # keep your original AeroCOPDNetLite implementation
        from .your_aerocpdnet_module import AeroCOPDNetLite  # adjust if needed
        return AeroCOPDNetLite(in_ch, n_classes, dropout)
    raise ValueError(f"Unknown model name: {name}")
