"""
Evaluation Metrics for Fingerprint Verification
------------------------------------------------
- Equal Error Rate (EER)
- ROC curve + AUC
- Cosine similarity threshold search
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
from typing import Tuple, Optional


def compute_similarities(
    emb1: torch.Tensor,
    emb2: torch.Tensor,
    metric: str = "cosine",
) -> np.ndarray:
    """
    Compute pairwise similarity between two sets of embeddings.
    emb1, emb2: (N, D) — already L2 normalized recommended for cosine.
    metric: 'cosine' or 'euclidean'
    """
    if metric == "cosine":
        sims = F.cosine_similarity(emb1, emb2, dim=1)
        return sims.cpu().numpy()
    else:
        dists = F.pairwise_distance(emb1, emb2)
        return -dists.cpu().numpy()   # negate so higher = more similar


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER) and the corresponding threshold.

    labels : 1 = same person, 0 = different
    scores : similarity scores (higher = more similar)
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    # Find the point where FPR ≈ FNR
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer      = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
    threshold = thresholds[eer_idx]
    return float(eer), float(threshold)


def evaluate_verification(
    model,
    dataloader,
    device: torch.device,
    metric: str = "cosine",
) -> dict:
    """
    Run full verification evaluation on a pair dataloader.

    Returns dict with keys: eer, eer_threshold, auc
    """
    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for img1, img2, labels in dataloader:
            img1, img2 = img1.to(device), img2.to(device)
            emb1, emb2 = model(img1, img2)
            sims = compute_similarities(emb1, emb2, metric=metric)
            all_scores.append(sims)
            all_labels.append(labels.numpy())

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)

    eer, eer_thresh = compute_eer(labels, scores)

    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    roc_auc = auc(fpr, tpr)

    return {
        "eer":           eer,
        "eer_threshold": eer_thresh,
        "auc":           roc_auc,
    }


def verify(
    emb1: torch.Tensor,
    emb2: torch.Tensor,
    threshold: float,
    metric: str = "cosine",
) -> bool:
    """
    Single-pair verification: returns True if same person.
    emb1, emb2: (1, D) or (D,) tensors
    """
    if emb1.dim() == 1:
        emb1, emb2 = emb1.unsqueeze(0), emb2.unsqueeze(0)
    sim = compute_similarities(emb1, emb2, metric=metric)[0]
    return bool(sim >= threshold)
