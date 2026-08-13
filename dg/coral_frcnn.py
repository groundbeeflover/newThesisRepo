from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from .coral_core import (
    FixedRandomProjection,
    covariance,
    mean_pairwise_coral,
    normalized_covariance_distance,
    sample_rows,
    zero_loss,
)


class FRCNNCoralLosses(nn.Module):
    """
    Five CORAL-style DG losses for the Faster R-CNN adaptation:

      img     : image-level cross-domain alignment
      ins     : instance-level cross-domain alignment
      cst     : image/instance covariance consistency
      ds_adv  : within-domain, across-class alignment
      ds_cls  : within-class, across-domain alignment

    The four alignment losses correspond to the four auxiliary GRL heads in
    Seemakurthy et al.; cst is the separate image/instance consistency term.
    """

    def __init__(
        self,
        image_dim: int = 256,
        roi_dim: int = 1024,
        common_dim: int = 256,
        max_spatial_samples_per_domain: int = 2048,
        max_roi_samples_per_group: int = 256,
        covariance_eps: float = 1e-5,
        projection_seed: int = 42,
    ):
        super().__init__()

        if common_dim != image_dim:
            self.image_projection = FixedRandomProjection(
                image_dim, common_dim, seed=projection_seed
            )
        else:
            self.image_projection = nn.Identity()

        self.roi_projection = FixedRandomProjection(
            roi_dim, common_dim, seed=projection_seed + 1
        )

        self.max_spatial_samples_per_domain = max_spatial_samples_per_domain
        self.max_roi_samples_per_group = max_roi_samples_per_group
        self.covariance_eps = covariance_eps

    @staticmethod
    def _valid_covariances(
        bank: Dict,
        keys: Iterable,
    ) -> List[torch.Tensor]:
        return [bank[k] for k in keys if k in bank and bank[k] is not None]

    @staticmethod
    def _pre_cap_counts(domains: torch.Tensor) -> Dict[int, int]:
        """Per-domain row counts *before* sample_rows() truncates to the cap.

        Used to check whether max_spatial_samples_per_domain /
        max_roi_samples_per_group are actually binding at a given
        samples-per-domain batch composition, or whether a single image
        already saturates them (see bs2-vs-bs8 sampling ablation notes).
        """
        if domains.numel() == 0:
            return {}
        return {
            int(d.item()): int((domains == d).sum().item())
            for d in torch.unique(domains)
        }

    @staticmethod
    def _alignment_or_zero(
        covs: List[torch.Tensor],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if len(covs) < 2:
            return zero_loss(reference)
        return mean_pairwise_coral(covs)

    def _image_samples(
        self,
        image_feature_map: torch.Tensor,
        image_domains: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert [B,C,H,W] into row-wise spatial samples [B*H*W,C].

        This keeps image-level CORAL usable with the paper batch size of 2:
        covariance is estimated from spatial activation vectors pooled over all
        images of each domain that are present in the minibatch.
        """
        if image_feature_map.ndim != 4:
            raise ValueError(
                f"Expected image feature map [B,C,H,W], got "
                f"{tuple(image_feature_map.shape)}"
            )

        b, c, h, w = image_feature_map.shape
        x = (
            image_feature_map
            .permute(0, 2, 3, 1)
            .reshape(b * h * w, c)
        )
        x = self.image_projection(x)

        d = (
            image_domains[:, None, None]
            .expand(b, h, w)
            .reshape(-1)
        )
        return x, d

    def _roi_metadata(
        self,
        roi_features: torch.Tensor,
        roi_labels_by_image: List[torch.Tensor],
        image_domains: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Concatenate per-image ROI labels and repeat each image domain by its
        actual number of sampled ROIs. No fixed 512-ROI assumption is made.
        """
        if len(roi_labels_by_image) != image_domains.numel():
            raise ValueError(
                "roi_labels_by_image length must equal number of image domains"
            )

        labels = torch.cat(
            [labels.to(roi_features.device) for labels in roi_labels_by_image],
            dim=0,
        )

        roi_domains = torch.cat(
            [
                image_domains[i].expand(labels_i.numel())
                for i, labels_i in enumerate(roi_labels_by_image)
            ],
            dim=0,
        ).to(roi_features.device)

        if labels.shape[0] != roi_features.shape[0]:
            raise RuntimeError(
                f"ROI metadata mismatch: {labels.shape[0]} labels for "
                f"{roi_features.shape[0]} feature vectors"
            )

        z = self.roi_projection(roi_features)
        return z, labels.long(), roi_domains.long()

    def _domain_covariance_bank(
        self,
        features: torch.Tensor,
        domains: torch.Tensor,
        max_samples: Optional[int],
    ) -> Dict[int, torch.Tensor]:
        bank: Dict[int, torch.Tensor] = {}

        for domain in torch.unique(domains):
            mask = domains == domain
            group = sample_rows(features[mask], max_samples)
            cov = covariance(group, eps=self.covariance_eps)
            if cov is not None:
                bank[int(domain.item())] = cov

        return bank

    def _conditional_covariance_bank(
        self,
        features: torch.Tensor,
        domains: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[Tuple[int, int], torch.Tensor]:
        bank: Dict[Tuple[int, int], torch.Tensor] = {}

        for domain in torch.unique(domains):
            d = int(domain.item())
            d_mask = domains == domain

            for cls in torch.unique(labels[d_mask]):
                c = int(cls.item())
                mask = d_mask & (labels == cls)
                group = sample_rows(
                    features[mask],
                    self.max_roi_samples_per_group,
                )
                cov = covariance(group, eps=self.covariance_eps)
                if cov is not None:
                    bank[(d, c)] = cov

        return bank

    def forward(
        self,
        image_feature_map: torch.Tensor,
        roi_features: torch.Tensor,
        roi_labels_by_image: List[torch.Tensor],
        image_domains: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            image_feature_map: [B, 256, H, W], e.g. FPN level '0'
            roi_features: [sum_i N_i, 1024], box-head output
            roi_labels_by_image: list of N_i ROI target-label tensors
            image_domains: [B]
        """
        image_domains = image_domains.to(
            device=roi_features.device, dtype=torch.long
        )

        image_x, image_d = self._image_samples(
            image_feature_map, image_domains
        )
        roi_x, roi_y, roi_d = self._roi_metadata(
            roi_features, roi_labels_by_image, image_domains
        )

        # Reused banks: each covariance is computed once.
        image_domain_cov = self._domain_covariance_bank(
            image_x,
            image_d,
            self.max_spatial_samples_per_domain,
        )
        roi_domain_cov = self._domain_covariance_bank(
            roi_x,
            roi_d,
            self.max_roi_samples_per_group,
        )
        conditional_cov = self._conditional_covariance_bank(
            roi_x,
            roi_d,
            roi_y,
        )

        present_domains = sorted(
            set(image_domain_cov.keys()) & set(roi_domain_cov.keys())
        )
        present_classes = sorted({c for (_, c) in conditional_cov.keys()})

        # 1) Image-level domain adversarial -> cross-domain CORAL
        loss_img = self._alignment_or_zero(
            [image_domain_cov[d] for d in present_domains],
            image_x,
        )

        # 2) Instance-level domain classification -> marginal ROI CORAL
        loss_ins = self._alignment_or_zero(
            [roi_domain_cov[d] for d in present_domains],
            roi_x,
        )

        # 3) Consistency: same-domain image/instance covariance geometry
        cst_terms = [
            normalized_covariance_distance(
                image_domain_cov[d], roi_domain_cov[d]
            )
            for d in present_domains
        ]
        loss_cst = (
            torch.stack(cst_terms).mean()
            if cst_terms
            else zero_loss(roi_x)
        )

        # 4) Domain-specific adversarial:
        #    same domain, align different classes (GWHD: BG vs FG)
        ds_adv_terms = []
        for d in present_domains:
            covs = [
                conditional_cov[(d, c)]
                for c in present_classes
                if (d, c) in conditional_cov
            ]
            if len(covs) >= 2:
                ds_adv_terms.append(mean_pairwise_coral(covs))

        loss_ds_adv = (
            torch.stack(ds_adv_terms).mean()
            if ds_adv_terms
            else zero_loss(roi_x)
        )

        # 5) Domain-specific classification:
        #    same class, align that class across domains
        ds_cls_terms = []
        for c in present_classes:
            covs = [
                conditional_cov[(d, c)]
                for d in present_domains
                if (d, c) in conditional_cov
            ]
            if len(covs) >= 2:
                ds_cls_terms.append(mean_pairwise_coral(covs))

        loss_ds_cls = (
            torch.stack(ds_cls_terms).mean()
            if ds_cls_terms
            else zero_loss(roi_x)
        )

        # Diagnostics for the bs2-vs-bs8 sampling ablation: pre-cap sample
        # counts per domain, so it's possible to tell directly (instead of
        # inferring) whether max_spatial_samples_per_domain /
        # max_roi_samples_per_group are binding, and whether foreground ROI
        # scarcity (not background) is the real limiter for ds_adv/ds_cls.
        # roi_y == 1 is the foreground/wheat-head class; 0 is background.
        fg_mask = roi_y == 1
        diag = {
            "num_domains_present": len(present_domains),
            "num_classes_present": len(present_classes),
            "image_spatial_counts": self._pre_cap_counts(image_d),
            "roi_counts": self._pre_cap_counts(roi_d),
            "roi_fg_counts": self._pre_cap_counts(roi_d[fg_mask]) if fg_mask.any() else {},
        }

        return {
            "img": loss_img,
            "ins": loss_ins,
            "cst": loss_cst,
            "ds_adv": loss_ds_adv,
            "ds_cls": loss_ds_cls,
            "diag": diag,
        }
