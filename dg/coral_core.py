from __future__ import annotations

from typing import Dict, Hashable, Iterable, Optional

import torch
import torch.nn as nn


def zero_loss(reference: torch.Tensor) -> torch.Tensor:
    """Differentiable scalar zero on the same device/dtype as reference."""
    return reference.sum() * 0.0


def sample_rows(x: torch.Tensor, max_samples: Optional[int]) -> torch.Tensor:
    """Randomly subsample rows without replacement."""
    if max_samples is None or x.shape[0] <= max_samples:
        return x
    idx = torch.randperm(x.shape[0], device=x.device)[:max_samples]
    return x[idx]


def covariance(x: torch.Tensor, eps: float = 0.0) -> Optional[torch.Tensor]:
    """
    Sample covariance of row-wise feature vectors.

    Args:
        x: [N, D]
        eps: optional diagonal regularizer

    Returns:
        [D, D] covariance matrix, or None when N < 2.
    """
    if x.ndim != 2:
        raise ValueError(f"Expected [N, D] features, got {tuple(x.shape)}")
    if x.shape[0] < 2:
        return None

    x = x.float()
    x = x - x.mean(dim=0, keepdim=True)
    cov = (x.transpose(0, 1) @ x) / float(x.shape[0] - 1)

    if eps > 0:
        cov = cov + eps * torch.eye(
            cov.shape[0], device=cov.device, dtype=cov.dtype
        )
    return cov


def deep_coral_from_covariances(
    cov_a: torch.Tensor,
    cov_b: torch.Tensor,
) -> torch.Tensor:
    """Standard Deep CORAL distance between two covariance matrices."""
    if cov_a.shape != cov_b.shape:
        raise ValueError(
            f"Covariance shapes must match, got {cov_a.shape} and {cov_b.shape}"
        )
    d = cov_a.shape[-1]
    return (cov_a - cov_b).pow(2).sum() / (4.0 * d * d)


def mean_pairwise_coral(covariances: Iterable[torch.Tensor]) -> torch.Tensor:
    """
    Mean Deep CORAL loss over all unordered pairs.

    Uses the covariance barycenter identity, avoiding explicit O(K^2) pair loops.
    The scaling is chosen so K=2 exactly equals standard pairwise Deep CORAL.
    """
    covs = list(covariances)
    if len(covs) < 2:
        raise ValueError("At least two covariance matrices are required")

    stacked = torch.stack(covs, dim=0)
    mean_cov = stacked.mean(dim=0, keepdim=True)
    k = stacked.shape[0]
    d = stacked.shape[-1]

    mean_sq_to_center = (
        (stacked - mean_cov)
        .pow(2)
        .sum(dim=(-2, -1))
        .mean()
    )

    # Average unordered pairwise squared distance:
    # 2K/(K-1) * mean_i ||Ci - Cbar||_F^2
    mean_pairwise_sq = (2.0 * k / float(k - 1)) * mean_sq_to_center
    return mean_pairwise_sq / (4.0 * d * d)


def normalized_covariance_distance(
    cov_a: torch.Tensor,
    cov_b: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Scale-invariant covariance distance, used for image/instance consistency.
    """
    if cov_a.shape != cov_b.shape:
        raise ValueError(
            f"Consistency covariances must share shape, got "
            f"{cov_a.shape} and {cov_b.shape}"
        )
    a = cov_a / cov_a.norm(p="fro").clamp_min(eps)
    b = cov_b / cov_b.norm(p="fro").clamp_min(eps)
    d = a.shape[-1]
    return (a - b).pow(2).sum() / float(d * d)


class FixedRandomProjection(nn.Module):
    """
    Non-trainable orthonormal random projection.

    Keeping the projection fixed prevents the auxiliary CORAL branch from
    learning a trivial zero projection while still allowing gradients to flow
    back into the detector features.
    """

    def __init__(self, in_dim: int, out_dim: int, seed: int = 42):
        super().__init__()
        if out_dim > in_dim:
            raise ValueError("out_dim must be <= in_dim")

        if in_dim == out_dim:
            weight = torch.eye(in_dim)
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            matrix = torch.randn(in_dim, out_dim, generator=generator)
            q, _ = torch.linalg.qr(matrix, mode="reduced")
            weight = q

        self.register_buffer("weight", weight, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.to(dtype=x.dtype)
