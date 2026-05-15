"""
Semantic Matching Module

This module implements the semantic matching components described in Sec. 4.2,
including timestep-adaptive conditioning with random labels and
semantic alignment loss computation.

The core idea is to adaptively inject noise into semantic embeddings based on
the diffusion timestep, while incorporating random labels to enhance
class diversity during generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def random_label_conditioning(t, y, random_label=0, t_upper=900, t_lower=500,
                               noise_weight=0.06, label_weight=0.01):
    """
    Timestep-adaptive conditioning with random label injection.
    Adaptively blends semantic embedding with noise and random label
    based on the diffusion timestep.

    Args:
        t: diffusion timestep tensor, shape (B,)
        y: semantic embedding (class condition), shape (B, D)
        random_label: random label signal for enhancing diversity (default: 0)
        t_upper: upper timestep threshold, above which semantic embedding is
                 fully replaced by noise and random label is disabled (default: 900)
        t_lower: lower timestep threshold, below which semantic embedding is
                 fully preserved (default: 500)
        noise_weight: scaling factor for injected random noise (default: 0.06)
        label_weight: scaling factor for random label (default: 0.01)

    Returns:
        y_renormalized: re-normalized conditioned embedding, shape (B, D)
    """
    device = y.device

    # Random noise with same shape as semantic embedding
    noise = torch.rand_like(y).to(device)

    # Determine interpolation coefficient and random label switch based on timestep
    # t > t_upper: lambda_t = 0, semantic info fully suppressed, random label disabled
    # t < t_lower: lambda_t = 1, semantic info fully preserved, random label enabled
    # t_lower <= t <= t_upper: linear interpolation, random label enabled
    if t[0] > t_upper:
        lambda_t = torch.tensor(0.0)
        use_random_label = 0
    elif t[0] < t_lower:
        lambda_t = torch.tensor(1.0)
        use_random_label = 1
    else:
        lambda_t = (t_upper - t[0]) / (t_upper - t_lower)
        use_random_label = 1

    # Blend semantic embedding with noise, and inject random label (Sec. 4.2)
    y_blended = (
        torch.sqrt(lambda_t) * y
        + noise_weight * torch.sqrt(1 - lambda_t) * noise
        + use_random_label * label_weight * torch.sqrt(1 - lambda_t) * random_label
    )

    # Re-normalize to match original semantic embedding's statistics
    mean_orig = torch.mean(y)
    std_orig = torch.std(y)
    mean_blended = torch.mean(y_blended)
    std_blended = torch.std(y_blended)

    y_renormalized = (y_blended - mean_blended) / std_blended * std_orig + mean_orig

    return y_renormalized