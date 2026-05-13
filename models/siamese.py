"""
Fingerprint Recognition Models
-------------------------------
1. FingerprintEncoder  - shared CNN backbone that maps image → embedding
2. SiameseNet          - wraps encoder for verification (contrastive loss)
3. TripletNet          - wraps encoder for triplet loss training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building Blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv → BN → ReLU → optional MaxPool"""

    def __init__(self, in_ch: int, out_ch: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class FingerprintEncoder(nn.Module):
    """
    Lightweight CNN that maps a grayscale fingerprint image to an L2-normalized
    embedding vector.

    Input : (B, 1, H, W)  — grayscale, H=W=96 default
    Output: (B, embed_dim)  — unit-norm embedding
    """

    def __init__(self, embed_dim: int = 128, image_size: int = 96):
        super().__init__()

        # ── Convolutional feature extractor ──────────────────────────────
        self.features = nn.Sequential(
            ConvBlock(1,   32, pool=True),   # → (B, 32,  48, 48)
            ConvBlock(32,  64, pool=True),   # → (B, 64,  24, 24)
            ConvBlock(64, 128, pool=True),   # → (B, 128, 12, 12)
            ConvBlock(128, 256, pool=True),  # → (B, 256,  6,  6)
            # Extra conv without pooling for richer features
            ConvBlock(256, 256, pool=False), # → (B, 256,  6,  6)
        )

        # Compute flattened size dynamically
        dummy = torch.zeros(1, 1, image_size, image_size)
        feat_size = self.features(dummy).view(1, -1).shape[1]

        # ── Embedding head ────────────────────────────────────────────────
        self.embed_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        emb  = self.embed_head(feat)
        return F.normalize(emb, p=2, dim=1)   # L2 normalize


# ---------------------------------------------------------------------------
# Siamese Network (Contrastive Loss)
# ---------------------------------------------------------------------------

class SiameseNet(nn.Module):
    """
    Takes two images → passes through shared encoder → returns both embeddings.
    Use with ContrastiveLoss.
    """

    def __init__(self, encoder: FingerprintEncoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, img1: torch.Tensor, img2: torch.Tensor):
        emb1 = self.encoder(img1)
        emb2 = self.encoder(img2)
        return emb1, emb2


# ---------------------------------------------------------------------------
# Triplet Network (Triplet Loss)
# ---------------------------------------------------------------------------

class TripletNet(nn.Module):
    """
    Takes (anchor, positive, negative) → returns three embeddings.
    Use with TripletMarginLoss.
    """

    def __init__(self, encoder: FingerprintEncoder):
        super().__init__()
        self.encoder = encoder

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ):
        ea = self.encoder(anchor)
        ep = self.encoder(positive)
        en = self.encoder(negative)
        return ea, ep, en


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------

class ContrastiveLoss(nn.Module):
    """
    L = (1-y) * 0.5 * d²  +  y * 0.5 * max(0, margin - d)²

    y=1 → different class (push apart)
    y=0 → same class    (pull together)
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor,
        label: torch.Tensor,   # 1=different, 0=same
    ) -> torch.Tensor:
        dist = F.pairwise_distance(emb1, emb2)
        same_loss = (1 - label) * 0.5 * dist.pow(2)
        diff_loss = label * 0.5 * F.relu(self.margin - dist).pow(2)
        return (same_loss + diff_loss).mean()


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def build_siamese(embed_dim: int = 128, image_size: int = 96) -> SiameseNet:
    enc = FingerprintEncoder(embed_dim=embed_dim, image_size=image_size)
    return SiameseNet(enc)


def build_triplet(embed_dim: int = 128, image_size: int = 96) -> TripletNet:
    enc = FingerprintEncoder(embed_dim=embed_dim, image_size=image_size)
    return TripletNet(enc)
