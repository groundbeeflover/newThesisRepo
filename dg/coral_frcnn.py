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
    """the five coral losses that replace the grl heads

      img     : line up domains at image level
      ins     : line up domains at instance level
      cst     : keep image and instance covariances consistent
      ds_adv  : same domain, line up the classes
      ds_cls  : same class, line up the domains

    the four alignment ones map onto the four grl heads in seemakurthy et al,
    cst is the separate consistency term
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
        """row counts per domain before sample_rows chops them down to the cap

        lets me see whether the caps are actually doing anything at a given
        batch composition or whether one image already blows past them
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
        """flatten [b,c,h,w] into rows of [b*h*w,c]

        this is what keeps image level coral working at the paper's batch size
        of 2, the covariance comes from spatial activations pooled over every
        image of a domain in the batch rather than from the 2 images themselves
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
        """glue the per image roi labels together and repeat each image's domain
        by however many rois it actually got, no assuming it's always 512
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
        """takes the fpn level '0' map [b, 256, h, w], the box head output
        [sum_i n_i, 1024], a list of roi label tensors per image, and the
        domain of each image [b]
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

        #work these out once and reuse them, the losses below all share them
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

        #1) image level, was the domain adversarial head
        loss_img = self._alignment_or_zero(
            [image_domain_cov[d] for d in present_domains],
            image_x,
        )

        #2) instance level, was the domain classification head
        loss_ins = self._alignment_or_zero(
            [roi_domain_cov[d] for d in present_domains],
            roi_x,
        )

        #3) consistency, image and instance covariance should look the same
        #   shape within a domain
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

        #4) domain specific adversarial, stay in one domain and line up the
        #   classes against each other (for gwhd that's just background vs wheat)
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

        #5) domain specific classification, pick a class and line it up across
        #   all the domains
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

        #sample counts before the caps kick in, so i can tell whether the caps
        #are actually binding and whether it's the foreground rois running out
        #rather than background that limits ds_adv and ds_cls
        #roi_y == 1 is wheat head, 0 is background
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
