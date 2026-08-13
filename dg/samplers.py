from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, Iterator, List

from torch.utils.data import Sampler


def _epoch_stats(domains: List[int], group_sizes: Dict[int, int],
                  image_counts: Dict[int, int], refill_counts: Dict[int, int]) -> List[dict]:
    """
    Shared helper for the bottleneck-isolation diagnostics (see thesis notes on
    CORAL bs2-vs-bs8 sampling ablation, Aug 2026).

    images_drawn / domain_size = how many times that domain's images were
    revisited this epoch in expectation terms; refill_count - 1 = how many
    times its pool was fully exhausted and reshuffled mid-epoch (0 means it
    never wrapped around). Both are logged per-domain so oversampling can be
    read off directly instead of inferred from batch_size alone.
    """
    rows = []
    for d in domains:
        size = group_sizes.get(d, 0)
        drawn = image_counts.get(d, 0)
        refills = refill_counts.get(d, 0)
        rows.append({
            "domain": d,
            "domain_size": size,
            "images_drawn": drawn,
            "times_recycled": max(refills - 1, 0),
            "oversample_ratio": (drawn / size) if size else float("nan"),
        })
    return rows


class DomainDiverseBatchSampler(Sampler[List[int]]):
    """
    Domain-balanced batch sampler that draws distinct domains within each batch.

    Designed for CORAL training where a cross-domain loss is zero unless at
    least two domains occur in the minibatch. Smaller domains are naturally
    oversampled because exhausted domain buckets are reshuffled and reused.

    Caveat (see DomainCappedBatchSampler below): this sampler ties the number
    of distinct domains in a batch to batch_size itself (one image per chosen
    domain, always). That means samples-per-domain is pinned at 1 no matter
    how large batch_size gets, so growing batch_size only adds more pairwise
    domain terms to the CORAL loss without ever improving the per-domain
    covariance estimates that loss depends on. This is a likely explanation
    for CORAL mAP@50 degrading at larger batch sizes; kept here for ablation
    comparisons against the alternatives below.
    """

    def __init__(
        self,
        domain_labels,
        batch_size: int,
        num_batches: int | None = None,
        seed: int = 42,
    ):
        if batch_size < 2:
            raise ValueError("CORAL training needs batch_size >= 2")

        #makes a dictionary of each sample image, identifying them by position in the dataset and grouping them by domain
        labels = [int(x) for x in domain_labels]
        self.groups = defaultdict(list)
        for idx, domain in enumerate(labels):
            self.groups[domain].append(idx)

        #orders the dictionary numerically by domain, makes it predictable. 
        self.domains = sorted(self.groups.keys())
        #throws error if there are less than two domains
        if len(self.domains) < 2:
            raise ValueError("At least two training domains are required")

        #throws error if the batch size is larger than the number of domains
        if batch_size > len(self.domains):
            raise ValueError(
                f"batch_size={batch_size} exceeds number of domains="
                f"{len(self.domains)} for distinct-domain batching"
            )

        self.batch_size = batch_size
        self.num_batches = (
            num_batches
            if num_batches is not None
            else len(labels) // batch_size
        )
        self.seed = seed
        self.epoch = 0
        self._image_counts: Dict[int, int] = {}
        self._refill_counts: Dict[int, int] = {}

    def __len__(self) -> int:
        return self.num_batches

    def get_last_epoch_stats(self) -> List[dict]:
        """Per-domain oversampling stats for the most recently completed epoch."""
        group_sizes = {d: len(self.groups[d]) for d in self.domains}
        return _epoch_stats(self.domains, group_sizes, self._image_counts, self._refill_counts)

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch) #uses seed and epoch as the random seed. for every epoch the collection is different, but every run is ultimately consistent
        self.epoch += 1

        self._image_counts = {d: 0 for d in self.domains}
        self._refill_counts = {d: 0 for d in self.domains}

        pools = {} #shuffled dictionary of entries per domain
        cursors = {} #pointer for each domain in the pool (index)

        #refills the indexes for one domain.
        def refill(domain: int):
            values = list(self.groups[domain])
            rng.shuffle(values)
            pools[domain] = values
            cursors[domain] = 0
            self._refill_counts[domain] += 1

        for domain in self.domains:
            refill(domain)

        for _ in range(self.num_batches):
            chosen_domains = rng.sample(self.domains, self.batch_size)
            batch = []

            for domain in chosen_domains:
                if cursors[domain] >= len(pools[domain]):
                    refill(domain)

                batch.append(pools[domain][cursors[domain]])
                cursors[domain] += 1
                self._image_counts[domain] += 1

            yield batch


class DomainCappedBatchSampler(Sampler[List[int]]):
    """
    Domain-balanced batch sampler that caps the number of *distinct* domains
    per batch at ``domains_per_batch`` (default 2) instead of tying it to
    batch_size.

    This mirrors the source/target pair structure used by Sun & Saenko's
    Deep CORAL paper and reference implementations: each batch is made of a
    small number of same-domain groups, each large enough to give a stable
    covariance estimate. Samples-per-domain = batch_size // domains_per_batch,
    so unlike DomainDiverseBatchSampler, per-domain sample count *grows*
    with batch_size instead of always being 1. Domain pairs (or k-tuples) are
    drawn at random each batch, so all C(num_domains, domains_per_batch)
    combinations get covered over the course of training rather than fixing
    one pair for a whole epoch as the original 2-domain UDA setting would.

    As in DomainDiverseBatchSampler, smaller domains are oversampled because
    exhausted domain buckets are reshuffled and reused; this is unavoidable
    with any domain-balanced sampler on an imbalanced dataset like GWHD.
    """

    def __init__(
        self,
        domain_labels,
        batch_size: int,
        domains_per_batch: int = 2,
        num_batches: int | None = None,
        seed: int = 42,
    ):
        if batch_size < 2:
            raise ValueError("CORAL training needs batch_size >= 2")
        if domains_per_batch < 2:
            raise ValueError("CORAL training needs domains_per_batch >= 2")
        if domains_per_batch > batch_size:
            raise ValueError(
                f"domains_per_batch={domains_per_batch} cannot exceed "
                f"batch_size={batch_size}"
            )

        labels = [int(x) for x in domain_labels]
        self.groups = defaultdict(list)
        for idx, domain in enumerate(labels):
            self.groups[domain].append(idx)

        self.domains = sorted(self.groups.keys())
        if len(self.domains) < domains_per_batch:
            raise ValueError(
                f"domains_per_batch={domains_per_batch} exceeds number of "
                f"training domains={len(self.domains)}"
            )

        self.batch_size = batch_size
        self.domains_per_batch = domains_per_batch
        self.num_batches = (
            num_batches
            if num_batches is not None
            else len(labels) // batch_size
        )
        self.seed = seed
        self.epoch = 0
        self._image_counts: Dict[int, int] = {}
        self._refill_counts: Dict[int, int] = {}

    def __len__(self) -> int:
        return self.num_batches

    def get_last_epoch_stats(self) -> List[dict]:
        """Per-domain oversampling stats for the most recently completed epoch."""
        group_sizes = {d: len(self.groups[d]) for d in self.domains}
        return _epoch_stats(self.domains, group_sizes, self._image_counts, self._refill_counts)

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1

        self._image_counts = {d: 0 for d in self.domains}
        self._refill_counts = {d: 0 for d in self.domains}

        pools = {}
        cursors = {}

        def refill(domain: int):
            values = list(self.groups[domain])
            rng.shuffle(values)
            pools[domain] = values
            cursors[domain] = 0
            self._refill_counts[domain] += 1

        for domain in self.domains:
            refill(domain)

        # Split batch_size evenly across the chosen domains; any remainder
        # (when batch_size doesn't divide evenly) goes to the first groups.
        base, remainder = divmod(self.batch_size, self.domains_per_batch)

        for _ in range(self.num_batches):
            chosen_domains = rng.sample(self.domains, self.domains_per_batch)
            batch = []

            for i, domain in enumerate(chosen_domains):
                n_take = base + (1 if i < remainder else 0)
                for _ in range(n_take):
                    if cursors[domain] >= len(pools[domain]):
                        refill(domain)
                    batch.append(pools[domain][cursors[domain]])
                    cursors[domain] += 1
                    self._image_counts[domain] += 1

            rng.shuffle(batch)
            yield batch


class NaturalDomainBatchSampler(Sampler[List[int]]):
    """
    Batch sampler with no artificial control over domains per batch at all:
    batches are ordinary random permutations of the full training set, the
    same behaviour as the plain ``shuffle=True`` DataLoader used for the
    non-DG baseline and for the original GRL script (DGOD/train_GWHD_dgfrcnn.py),
    which does not do any domain-aware batching either.

    Domain composition of a batch is left entirely to chance: large domains
    (e.g. ETHZ_1 with 747 images) show up in most batches, tiny ones (e.g.
    Arvalis_8 with 20 images) rarely do, and samples-per-domain scales
    naturally with batch_size instead of being fixed. This is the closest
    thing to "not fixing # of domains to batch size."

    The only safeguard added is a minimum of two distinct domains per batch:
    FRCNNCoralLosses returns a zero CORAL loss whenever a batch is
    single-domain (see FRCNNCoralLosses._alignment_or_zero), so a purely
    unconstrained sampler would occasionally waste a training step's CORAL
    signal. When ``ensure_min_domains=True`` (default), a single element of
    an accidentally single-domain batch is swapped for a random sample from
    a different domain. Set it to False to reproduce the baseline's batching
    behaviour exactly, with no CORAL-specific guarantee at all.
    """

    def __init__(
        self,
        domain_labels,
        batch_size: int,
        num_batches: int | None = None,
        seed: int = 42,
        ensure_min_domains: bool = True,
    ):
        if batch_size < 2:
            raise ValueError("CORAL training needs batch_size >= 2")

        labels = [int(x) for x in domain_labels]
        self.labels = labels
        self.n = len(labels)

        self.by_domain = defaultdict(list)
        for idx, domain in enumerate(labels):
            self.by_domain[domain].append(idx)

        self.domains = sorted(self.by_domain.keys())
        if len(self.domains) < 2:
            raise ValueError("At least two training domains are required")

        self.batch_size = batch_size
        self.num_batches = (
            num_batches
            if num_batches is not None
            else self.n // batch_size
        )
        self.seed = seed
        self.epoch = 0
        self.ensure_min_domains = ensure_min_domains
        self._image_counts: Dict[int, int] = {}
        self._wraparound_count = 0

    def __len__(self) -> int:
        return self.num_batches

    def get_last_epoch_stats(self) -> List[dict]:
        """
        Per-domain draw stats for the most recently completed epoch. Unlike
        the two domain-balanced samplers, there's no per-domain pool/refill
        here -- "times_recycled" is reported as the single shared full-dataset
        wraparound count (same for every domain, since it's one global
        shuffled index list, not per-domain pools).
        """
        group_sizes = {d: len(self.by_domain[d]) for d in self.domains}
        refill_counts = {d: self._wraparound_count for d in self.domains}
        return _epoch_stats(self.domains, group_sizes, self._image_counts, refill_counts)

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1

        self._image_counts = {d: 0 for d in self.domains}
        self._wraparound_count = 0

        order = list(range(self.n))
        rng.shuffle(order)
        cursor = 0

        for _ in range(self.num_batches):
            if cursor + self.batch_size > len(order):
                # Wrap around with a fresh shuffle, same spirit as a new
                # DataLoader epoch, so we never run out of data mid-batch.
                order = list(range(self.n))
                rng.shuffle(order)
                cursor = 0
                self._wraparound_count += 1

            batch = order[cursor: cursor + self.batch_size]
            cursor += self.batch_size

            if self.ensure_min_domains:
                present = {self.labels[i] for i in batch}
                if len(present) < 2:
                    only_domain = next(iter(present))
                    other_domain = rng.choice(
                        [d for d in self.domains if d != only_domain]
                    )
                    batch[0] = rng.choice(self.by_domain[other_domain])

            for idx in batch:
                self._image_counts[self.labels[idx]] += 1

            yield batch
