"""
Inference — Fingerprint Verification
======================================
Given two fingerprint images, decide if they belong to the same person.

Usage:
  python inference.py --img1 finger_a.BMP --img2 finger_b.BMP \
                      --checkpoint checkpoints/encoder_best.pt \
                      --threshold 0.5
"""

import argparse
import torch
from pathlib import Path
from PIL import Image
import sys

sys.path.insert(0, str(Path(__file__).parent))

from models.siamese import FingerprintEncoder
from data.dataset import get_transforms
from utils.metrics import verify, compute_similarities


def load_encoder(checkpoint_path: str, device: torch.device) -> FingerprintEncoder:
    ckpt = torch.load(checkpoint_path, map_location=device)
    encoder = FingerprintEncoder(
        embed_dim=ckpt["embed_dim"],
        image_size=ckpt["image_size"],
    )
    encoder.load_state_dict(ckpt["state_dict"])
    encoder.eval().to(device)
    return encoder


def embed_image(encoder, image_path: str, device: torch.device, image_size: int = 96):
    transform = get_transforms(image_size, train=False)
    img = Image.open(image_path).convert("L")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        return encoder(tensor)


def main():
    p = argparse.ArgumentParser("Fingerprint Verifier")
    p.add_argument("--img1",       required=True)
    p.add_argument("--img2",       required=True)
    p.add_argument("--checkpoint", default="checkpoints/encoder_best.pt")
    p.add_argument("--threshold",  type=float, default=0.7,
                   help="Cosine similarity threshold (0–1). Higher = stricter.")
    p.add_argument("--image_size", type=int,   default=96,
                   help="Must match the image_size used during training.")
    args = p.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = load_encoder(args.checkpoint, device)

    emb1 = embed_image(encoder, args.img1, device, image_size=args.image_size)
    emb2 = embed_image(encoder, args.img2, device, image_size=args.image_size)

    similarity = compute_similarities(emb1, emb2, metric="cosine")[0]
    match      = similarity >= args.threshold

    print(f"\n  Image 1    : {args.img1}")
    print(f"  Image 2    : {args.img2}")
    print(f"  Similarity : {similarity:.4f}  (threshold={args.threshold})")
    print(f"  Decision   : {'✅ SAME PERSON' if match else '❌ DIFFERENT PERSON'}\n")


if __name__ == "__main__":
    main()
