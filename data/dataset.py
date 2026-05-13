"""
SOCOFing Dataset Loader
-----------------------
SOCOFing structure:
  Real/          ← original fingerprints
    001__M_Left_index_finger.BMP
    001__M_Left_little_finger.BMP
    ...
  Altered/
    Altered-Easy/
    Altered-Medium/
    Altered-Hard/

Each filename encodes: <SubjectID>__<Gender>_<Hand>_<Finger>_finger[_obliterated|_zcut|_cr].BMP
Subject IDs: 1–600  (600 subjects × 10 fingers = 6000 original images)
"""

import os
import re
import random
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_subject_id(filename: str) -> int:
    """Extract subject ID from SOCOFing filename."""
    match = re.match(r"(\d+)__", os.path.basename(filename))
    return int(match.group(1)) if match else -1


def load_image_paths(root_dir: str, use_altered: bool = False) -> List[str]:
    """Return list of absolute image paths."""
    root = Path(root_dir)
    paths = []

    real_dir = root / "SOCOFing" / "Real"
    if not real_dir.exists():
        # Try direct structure
        real_dir = root / "Real"
    if real_dir.exists():
        paths += [str(p) for p in real_dir.rglob("*.BMP")]

    if use_altered:
        for difficulty in ["Altered-Easy", "Altered-Medium", "Altered-Hard"]:
            alt_dir = root / "SOCOFing" / "Altered" / difficulty
            if not alt_dir.exists():
                alt_dir = root / "Altered" / difficulty
            if alt_dir.exists():
                paths += [str(p) for p in alt_dir.rglob("*.BMP")]

    return paths


def build_subject_map(paths: List[str]) -> dict:
    """Map subject_id -> [list of image paths]."""
    subject_map = {}
    for p in paths:
        sid = parse_subject_id(p)
        subject_map.setdefault(sid, []).append(p)
    return subject_map


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_transforms(image_size: int = 96, train: bool = True):
    """Return torchvision transform pipeline."""
    base = [
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=1),
    ]
    if train:
        augment = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        ]
    else:
        augment = []

    normalize = [
        transforms.ToTensor(),                          # [0,1]
        transforms.Normalize(mean=[0.5], std=[0.5]),    # [-1,1]
    ]
    return transforms.Compose(base + augment + normalize)


# ---------------------------------------------------------------------------
# Pair Dataset (for Siamese / Contrastive training)
# ---------------------------------------------------------------------------

class FingerprintPairDataset(Dataset):
    """
    Returns pairs (img1, img2, label) where:
      label=1 → same subject (positive pair)
      label=0 → different subjects (negative pair)

    Pairs are generated on-the-fly for unlimited variety.
    """

    def __init__(
        self,
        root_dir: str,
        num_pairs: int = 10_000,
        train: bool = True,
        image_size: int = 96,
        use_altered: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        self.transform = get_transforms(image_size, train)
        random.seed(seed)

        all_paths = load_image_paths(root_dir, use_altered)
        if not all_paths:
            raise FileNotFoundError(
                f"No .BMP images found under {root_dir}. "
                "Run data/download_dataset.py first."
            )

        self.subject_map = build_subject_map(all_paths)
        self.subjects = [s for s, imgs in self.subject_map.items() if len(imgs) >= 2]

        print(f"  Loaded {len(all_paths)} images | {len(self.subjects)} subjects")

        self.pairs: List[Tuple[str, str, int]] = []
        self._generate_pairs(num_pairs)

    def _generate_pairs(self, num_pairs: int):
        n_pos = num_pairs // 2
        n_neg = num_pairs - n_pos

        # Positive pairs
        for _ in range(n_pos):
            sid = random.choice(self.subjects)
            img1, img2 = random.sample(self.subject_map[sid], 2)
            self.pairs.append((img1, img2, 1))

        # Negative pairs
        for _ in range(n_neg):
            sid1, sid2 = random.sample(self.subjects, 2)
            img1 = random.choice(self.subject_map[sid1])
            img2 = random.choice(self.subject_map[sid2])
            self.pairs.append((img1, img2, 0))

        random.shuffle(self.pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path1, path2, label = self.pairs[idx]
        img1 = self.transform(Image.open(path1).convert("L"))
        img2 = self.transform(Image.open(path2).convert("L"))
        return img1, img2, torch.tensor(label, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Triplet Dataset (for Triplet Loss training)
# ---------------------------------------------------------------------------

class FingerprintTripletDataset(Dataset):
    """
    Returns (anchor, positive, negative) triplets.
    Anchor & positive → same subject
    Negative → different subject
    """

    def __init__(
        self,
        root_dir: str,
        num_triplets: int = 10_000,
        train: bool = True,
        image_size: int = 96,
        use_altered: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        self.transform = get_transforms(image_size, train)
        random.seed(seed)

        all_paths = load_image_paths(root_dir, use_altered)
        self.subject_map = build_subject_map(all_paths)
        self.subjects = [s for s, imgs in self.subject_map.items() if len(imgs) >= 2]

        print(f"  Loaded {len(all_paths)} images | {len(self.subjects)} subjects")

        self.triplets: List[Tuple[str, str, str]] = []
        self._generate_triplets(num_triplets)

    def _generate_triplets(self, num_triplets: int):
        for _ in range(num_triplets):
            anchor_sid = random.choice(self.subjects)
            neg_sid = random.choice([s for s in self.subjects if s != anchor_sid])

            anchor, positive = random.sample(self.subject_map[anchor_sid], 2)
            negative = random.choice(self.subject_map[neg_sid])
            self.triplets.append((anchor, positive, negative))

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        a_path, p_path, n_path = self.triplets[idx]
        anchor   = self.transform(Image.open(a_path).convert("L"))
        positive = self.transform(Image.open(p_path).convert("L"))
        negative = self.transform(Image.open(n_path).convert("L"))
        return anchor, positive, negative
