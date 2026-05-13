"""
Training Script — Fingerprint Recognition
==========================================
Supports two training modes:
  --mode contrastive   →  Siamese + ContrastiveLoss
  --mode triplet       →  TripletNet + TripletMarginLoss

Usage:
  python train.py --data_dir ./data/raw --mode triplet --epochs 30
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# ── Local imports ────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import FingerprintPairDataset, FingerprintTripletDataset
from models.siamese import (
    build_siamese, build_triplet,
    ContrastiveLoss, FingerprintEncoder,
)
from utils.metrics import evaluate_verification


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("Fingerprint Recognition Trainer")
    p.add_argument("--data_dir",   default="./data/raw",   help="Root folder of SOCOFing dataset")
    p.add_argument("--mode",       default="triplet",      choices=["contrastive", "triplet"])
    p.add_argument("--embed_dim",  type=int, default=128)
    p.add_argument("--image_size", type=int, default=96)
    p.add_argument("--epochs",     type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--num_pairs",  type=int, default=20_000, help="Pairs/triplets per epoch build")
    p.add_argument("--margin",     type=float, default=1.0,  help="Loss margin")
    p.add_argument("--use_altered",action="store_true",     help="Include altered fingerprint images")
    p.add_argument("--save_dir",   default="./checkpoints")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_contrastive(args, device):
    print("\n── Contrastive (Siamese) Training ─────────────────────────────")

    print("Building dataset...")
    full_ds = FingerprintPairDataset(
        root_dir=args.data_dir,
        num_pairs=args.num_pairs,
        train=True,
        image_size=args.image_size,
        use_altered=args.use_altered,
    )
    val_size  = int(0.1 * len(full_ds))
    train_ds, val_ds = random_split(full_ds, [len(full_ds) - val_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model     = build_siamese(args.embed_dim, args.image_size).to(device)
    criterion = ContrastiveLoss(margin=args.margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for img1, img2, labels in train_loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            emb1, emb2 = model(img1, img2)
            loss = criterion(emb1, emb2, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ── Validate ──────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img1, img2, labels in val_loader:
                img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
                emb1, emb2 = model(img1, img2)
                val_loss += criterion(emb1, emb2, labels).item()
        val_loss /= len(val_loader)

        scheduler.step()

        elapsed = time.time() - t0
        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f} | {elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save(model.encoder, args, "encoder_best.pt")

    _save(model.encoder, args, "encoder_final.pt")


def train_triplet(args, device):
    print("\n── Triplet Training ────────────────────────────────────────────")

    print("Building dataset...")
    full_ds = FingerprintTripletDataset(
        root_dir=args.data_dir,
        num_triplets=args.num_pairs,
        train=True,
        image_size=args.image_size,
        use_altered=args.use_altered,
    )
    val_size  = int(0.1 * len(full_ds))
    train_ds, val_ds = random_split(full_ds, [len(full_ds) - val_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model     = build_triplet(args.embed_dim, args.image_size).to(device)
    criterion = nn.TripletMarginLoss(margin=args.margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for anchor, positive, negative in train_loader:
            anchor   = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            ea, ep, en = model(anchor, positive, negative)
            loss = criterion(ea, ep, en)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ── Validate ──────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for anchor, positive, negative in val_loader:
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                ea, ep, en = model(anchor, positive, negative)
                val_loss += criterion(ea, ep, en).item()
        val_loss /= len(val_loader)

        scheduler.step()

        elapsed = time.time() - t0
        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f} | {elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save(model.encoder, args, "encoder_best.pt")

    _save(model.encoder, args, "encoder_final.pt")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(encoder: FingerprintEncoder, args, filename: str):
    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, filename)
    torch.save({
        "state_dict": encoder.state_dict(),
        "embed_dim":  args.embed_dim,
        "image_size": args.image_size,
    }, path)
    print(f"  💾 Saved → {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.mode == "contrastive":
        train_contrastive(args, device)
    else:
        train_triplet(args, device)
