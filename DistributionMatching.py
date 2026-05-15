
"""
Distribution Matching Module for DMGD Framework.

This module implements the Distribution Matching component (Sec. 4.3) of the
Dual Matching Guided Diffusion (DMGD) framework. It includes:
  - Distribution Approximate Matching via K-means (Eq. 16)
  - Optimal Transport (Sinkhorn) loss computation (Eq. 12)
  - Greedy Progressive Matching support (Eq. 18)
  - Utility functions for latent feature extraction and preprocessing

Reference: "DMGD: Train-Free Dataset Distillation with Semantic-Distribution
Matching in Diffusion Models"
"""

import sys
import math
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import numpy as np
from collections import OrderedDict, defaultdict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

from data import ImageFolder
from models import DiT_models
from download import find_model
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
import torch.nn.functional as F
from fast_pytorch_kmeans import KMeans


# =============================================================================
# Model Utilities
# =============================================================================

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Update the Exponential Moving Average (EMA) model parameters
    towards the current model parameters.

    Args:
        ema_model: The EMA shadow model.
        model: The current training model.
        decay: EMA decay rate (default: 0.9999).
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.

    Args:
        model: The target model.
        flag: Whether to enable gradient computation (default: True).
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    Terminate Distributed Data Parallel (DDP) training process group.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to both a log file and stdout.
    Only rank 0 performs actual logging in distributed settings.

    Args:
        logging_dir: Directory path for log file output.

    Returns:
        logger: Configured logging.Logger instance.
    """
    if dist.get_rank() == 0:
        logger = logging.getLogger(__name__)
        logger.propagate = False
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        if logging_dir:
            fh = logging.FileHandler(f'{logging_dir}/logs.txt')
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def mark_difffit_trainable(model, is_bitfit=False):
    """
    Mark parameters for DiffFit fine-tuning by selectively enabling gradients.

    Args:
        model: The diffusion model to configure.
        is_bitfit: If True, only bias terms are trainable (default: False).

    Returns:
        model: Model with updated requires_grad flags.
    """
    if is_bitfit:
        trainable_names = ['bias']
    else:
        trainable_names = ["bias", "norm", "gamma", "y_embed"]

    for par_name, par_tensor in model.named_parameters():
        par_tensor.requires_grad = any([kw in par_name for kw in trainable_names])
    return model


# =============================================================================
# Mathematical Utilities
# =============================================================================

def l1_normalize(tensor):
    """
    Perform L1 normalization on a tensor so that elements sum to 1.
    Used to normalize mass coefficients m_i in the approximate distribution (Eq. 16).

    Args:
        tensor: Input tensor (e.g., cluster cardinalities from K-means).

    Returns:
        Normalized tensor with elements summing to 1.
    """
    total_sum = torch.sum(tensor)
    if total_sum == 0:
        return torch.zeros_like(tensor)
    normalized_tensor = tensor / total_sum
    return normalized_tensor


def map_to_hypersphere(vectors):
    """
    Project vectors onto the unit hypersphere via L2 normalization.
    Used to define the distribution space with hyperspherical projection (Sec. 4.3).

    Args:
        vectors: Input tensor of shape (N, D).

    Returns:
        Normalized vectors on the unit hypersphere, shape (N, D).
    """
    norms = torch.norm(vectors, p=2, dim=1, keepdim=True)
    return vectors / norms


def cosine_similarity(tensor_a, tensor_b):
    """
    Compute pairwise cosine similarity between two sets of vectors.

    Args:
        tensor_a: First set of vectors, shape (N, D).
        tensor_b: Second set of vectors, shape (M, D).

    Returns:
        Cosine similarity matrix of shape (N, M).
    """
    n, m = tensor_a.shape[0], tensor_b.shape[0]
    dot_product = torch.matmul(tensor_a, tensor_b.T)
    norm_a = torch.norm(tensor_a, dim=-1).view(n, 1).repeat(1, m)
    norm_b = torch.norm(tensor_b, dim=-1).view(1, m).repeat(n, 1)
    return dot_product / (norm_a * norm_b)


# =============================================================================
# Image Preprocessing
# =============================================================================

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM (Ablated Diffusion Model).
    Reference: https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126

    Performs progressive BOX downsampling followed by BICUBIC resize and center crop.

    Args:
        pil_image: Input PIL Image.
        image_size: Target square crop size.

    Returns:
        Center-cropped PIL Image of size (image_size, image_size).
    """
    # Progressive BOX downsampling to improve quality
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


# =============================================================================
# EMA Feature Tracker (for Mean Matching baseline, Proposition 2)
# =============================================================================

class EMAFeatureTracker:
    """
    Exponential Moving Average tracker for class-wise latent features.

    Used to compute the mean representation of each class in latent space.
    This corresponds to the mean matching approximation P_T^(1) = delta_mu
    mentioned in Proposition 2 as a special case of distribution approximation.

    Args:
        alpha: EMA smoothing coefficient in (0, 1). Lower values give more
               weight to historical observations.
    """

    def __init__(self, alpha):
        self.alpha = alpha
        self.class_ema_dict = {}

    def update(self, class_id, value):
        """
        Update the EMA value for a given class.

        Args:
            class_id: Integer class identifier.
            value: Current feature tensor for this class.
        """
        if class_id not in self.class_ema_dict:
            self.class_ema_dict[class_id] = value
        else:
            prev_ema = self.class_ema_dict[class_id]
            self.class_ema_dict[class_id] = (
                self.alpha * value + (1 - self.alpha) * prev_ema
            )

    def get(self, class_id):
        """
        Retrieve the EMA value for a given class.

        Args:
            class_id: Integer class identifier.

        Returns:
            EMA feature tensor for the specified class, or None if not found.
        """
        return self.class_ema_dict.get(class_id)

    def get_all(self):
        """
        Retrieve all class EMA values.

        Returns:
            Dictionary mapping class_id -> EMA feature tensor.
        """
        return self.class_ema_dict


# =============================================================================
# Distribution Approximate Matching (Sec. 4.3, Definition 1, Eq. 16)
# =============================================================================

def kmeans_approximate_distribution(data, num_support_points):
    """
    Compute K-means based discrete approximate distribution P_tilde_T (Eq. 16).

    This implements the Distribution Approximate Matching strategy that finds
    K support points (cluster centroids) and corresponding mass coefficients
    (cluster cardinalities) to approximate the target distribution while
    preserving its geometric structure.

    The discrete approximation is defined as:
        P_tilde_T = sum_{i=1}^{K} m_i * delta_{k_i}
    where k_i are cluster centroids and m_i = c_i / sum(c_j) are normalized
    cluster cardinalities.

    Args:
        data: Target class feature tensor of shape (N, D) in latent space.
        num_support_points: Number of support points K for approximation.

    Returns:
        Dictionary containing:
            'data': Support points (cluster centroids), shape (K, D).
            'P': Mass coefficients (cluster cardinalities), shape (K,).
                 These are raw counts; L1 normalization is applied during OT computation.
    """
    kmeans = KMeans(n_clusters=num_support_points, mode='euclidean')
    cluster_labels = kmeans.fit_predict(data)
    support_points = kmeans.centroids
    mass_coefficients = torch.bincount(cluster_labels)
    return {"data": support_points, "P": mass_coefficients}


def density_based_sampling(data, num_support_points):
    """
    Alternative distribution approximation via density-weighted random sampling.

    Samples representative points proportional to local density, providing
    another approach to the discrete distribution approximation problem (Def. 1).

    Args:
        data: Target class feature tensor of shape (N, D).
        num_support_points: Number of support points to sample.

    Returns:
        Dictionary containing:
            'data': Sampled support points, shape (num_support_points, D).
            'P': Sampling probabilities for each selected point.
    """
    distances = torch.cdist(data, data)
    # Density estimate: inverse of sum of distances to all other points
    densities = 1.0 / (torch.sum(distances, dim=1) + 1e-8)
    probabilities = densities / torch.sum(densities)
    indices = torch.multinomial(probabilities, num_support_points, replacement=False)
    return {"data": data[indices], "P": probabilities[indices]}


# =============================================================================
# Latent Feature Extraction
# =============================================================================

def compute_class_mean_features(
    image_size=256,
    data_path="data/dataset/train",
    spec='woof',
    phase=0,
    nclass=10,
    num_workers=4,
    global_batch_size=64,
    ipc=-1
):
    """
    Compute class-wise mean latent features using EMA aggregation.

    This corresponds to the mean matching approximation P_tilde_T^(1) = delta_mu
    (Proposition 2), which serves as a baseline for distribution approximation.

    Args:
        image_size: Target image resolution for preprocessing.
        data_path: Path to the target dataset.
        spec: Dataset specification ('woof', 'nette', etc.).
        phase: Dataset phase/split indicator.
        nclass: Number of classes.
        num_workers: DataLoader worker count.
        global_batch_size: Batch size for feature extraction.
        ipc: Instances per class (-1 for all available).

    Returns:
        Dictionary mapping class_id -> mean latent feature tensor.
    """
    device = 'cuda'
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(
        data_path, transform=transform, nclass=nclass,
        ipc=ipc, spec=spec, phase=phase,
        seed=0, return_origin=True
    )
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path="pretrained_models/"
    ).to(device)
    vae.eval()

    loader = DataLoader(
        dataset,
        batch_size=global_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    feature_tracker = EMAFeatureTracker(alpha=0.2)
    for x, _, y in loader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            # Encode images into VAE latent space and scale
            x = vae.encode(x).latent_dist.sample().mul_(0.18215)

        # Update EMA for each class present in the batch
        unique_classes = set(y.tolist())
        for class_id in unique_classes:
            class_mask = (y == class_id)
            class_mean = torch.mean(
                x[class_mask].detach(), dim=0
            ).unsqueeze(0).flatten(1)
            feature_tracker.update(class_id=class_id, value=class_mean)

    return feature_tracker.get_all()


def compute_approximate_distribution(
    image_size=256,
    data_path="data/dataset/train",
    spec='woof',
    phase=0,
    nclass=10,
    num_workers=4,
    global_batch_size=64,
    ipc=-1,
    K_num=10
):
    """
    Compute K-means based approximate distribution for all classes (Eq. 16).

    This is the main Distribution Approximate Matching function that:
    1. Extracts latent features for all samples using the VAE encoder.
    2. Performs class-wise K-means clustering to obtain support points {k_i}
       and mass coefficients {m_i} for the discrete approximation P_tilde_T.

    The resulting approximate distribution satisfies W(P_tilde_T, P_T) <= epsilon
    (Corollary 1), enabling efficient OT computation during guided sampling.

    Args:
        image_size: Target image resolution for preprocessing.
        data_path: Path to the target dataset.
        spec: Dataset specification ('woof', 'nette', etc.).
        phase: Dataset phase/split indicator.
        nclass: Number of classes.
        num_workers: DataLoader worker count.
        global_batch_size: Batch size for feature extraction.
        ipc: Instances per class (-1 for all available).
        K_num: Number of support points K per class (default: 10).

    Returns:
        Dictionary mapping class_id -> {
            'data': Support points (K cluster centroids), shape (K, D).
            'P': Mass coefficients (cluster cardinalities), shape (K,).
        }
    """
    print(f"Computing approximate distribution: spec={spec}, nclass={nclass}, phase={phase}")
    device = 'cuda'

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(
        data_path, transform=transform, nclass=nclass,
        ipc=ipc, spec=spec, phase=phase,
        seed=0, return_origin=True
    )

    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path="pretrained_models/"
    ).to(device)
    vae.eval()

    loader = DataLoader(
        dataset,
        batch_size=global_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    # Step 1: Extract latent features for all samples, grouped by class
    class_latent_vectors = defaultdict(list)
    with torch.no_grad():
        for x, _, y in loader:
            x = x.to(device)
            y = y.to(device)
            # Encode images into VAE latent space and scale
            x = vae.encode(x).latent_dist.sample().mul_(0.18215)

            for i in range(len(y)):
                class_latent_vectors[y[i].item()].append(
                    x[i].unsqueeze(0).flatten(1)
                )

    print(f"Extracted features for classes: {sorted(class_latent_vectors.keys())}")

    # Step 2: Perform class-wise K-means to obtain approximate distribution
    start_time = time()
    approximate_distribution = {}
    for class_id, vectors in class_latent_vectors.items():
        # Concatenate all latent vectors for this class: shape (N_class, D)
        class_features = torch.cat(vectors, dim=0)
        # Apply K-means clustering to find support points and mass coefficients
        approximate_distribution[class_id] = kmeans_approximate_distribution(
            class_features, K_num
        )
    elapsed_time = time() - start_time
    print(f"Distribution approximation completed in {elapsed_time:.4f} seconds")

    return approximate_distribution


# =============================================================================
# Optimal Transport Loss (Sec. 4.3, Eq. 12-13)
# =============================================================================

def compute_ot_distribution_matching_loss(x, t, y, approximate_distribution, memory_samples=None, epsilon=0.1, num_iterations=5, **kwargs):
    """
    Compute Optimal Transport distribution matching loss via Sinkhorn iteration.
    Corresponds to Eq. 12 in the paper.

    Args:
        x: current batch samples, shape (B, C, ...) 
        t: timestep (unused in this function but kept for interface consistency)
        y: class label tensor
        cfg_scale: classifier-free guidance scale (unused here)
        approximate_distribution: dict containing per-class approximate distributions
        memory_samples: list of historical memory samples
        epsilon: entropic regularization coefficient
        num_iterations: number of Sinkhorn iterations
    """
    # Get target distribution embeddings (K centroids from Eq. 16)
    pos_embeddings = approximate_distribution[y[0].item()]["data"]

    # Combine memory samples with current batch as source distribution
    if len(memory_samples) > 0:
        neg_embeddings = torch.cat(memory_samples).flatten(start_dim=1)
        all_embeddings = torch.cat((neg_embeddings, x.flatten(start_dim=1)))
    else:
        all_embeddings = x.flatten(start_dim=1)

    # Project both distributions onto hypersphere
    all_embeddings = map_to_hypersphere(all_embeddings)
    pos_embeddings = map_to_hypersphere(pos_embeddings)

    # Compute cost matrix (pairwise distances)
    C = torch.cdist(all_embeddings, pos_embeddings)  # shape: (N, K)

    # Source distribution: uniform weights
    N = all_embeddings.shape[0]
    a = (torch.ones(N) / N).to('cuda').reshape(-1, 1)  # shape: (N, 1)

    # Target distribution: weights from K-means (Eq. 16)
    b = l1_normalize(approximate_distribution[y[0].item()]['P']).to('cuda')  # shape: (K,)

    # Gibbs kernel
    K = torch.exp(-C / epsilon)  # shape: (N, K)

    # Sinkhorn-Knopp matrix scaling iterations
    for _ in range(num_iterations):
        # Row normalization: enforce source marginal
        K = K * (a / torch.sum(K, dim=1, keepdim=True))
        # Column normalization: enforce target marginal
        K = K * (b / torch.sum(K, dim=0, keepdim=True))

    # Optimal transport plan
    P = K

    # Compute transport cost (Eq. 12)
    loss = torch.sum(P * C)

    return loss


# =============================================================================
# Entry Point for Testing
# =============================================================================

if __name__ == "__main__":
    # Test: compute mean features for verification
    class_mean_features = compute_class_mean_features()
    print(class_mean_features[193])