from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, Iterator, List

from torch.utils.data import Sampler


def _epoch_stats(domains: List[int], group_sizes: Dict[int, int],
                  image_counts: Dict[int, int], refill_counts: Dict[int, int]) -> List[dict]:
    """shared bit for the sampling diagnostics, used by all three samplers

    images_drawn / domain_size is roughly how many times a domain got revisited
    this epoch, and times_recycled is how often its pool ran dry and got
    reshuffled (0 means it never wrapped), logging both per domain so i can just
    read the oversampling off instead of guessing it from batch_size
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
    """every image in a batch comes from a different domain

    coral needs at least two domains in a batch or the loss is just zero, so
    this guarantees it, small domains end up oversampled because their buckets
    get reshuffled and reused once they run out

    the annoying part (which is why DomainCappedBatchSampler exists): this ties
    the number of domains to batch_size, one image per domain always, so
    samples-per-domain is stuck at 1 forever, and bumping batch_size just adds
    more pairwise terms without making any of the per-domain covariances any
    better, probably why coral map@50 gets worse at bigger batch sizes, keeping
    it around for the ablation
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

        #orders the dictionary numerically by domain, makes it predictable
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
        """per domain oversampling stats for whatever epoch just finished"""
        group_sizes = {d: len(self.groups[d]) for d in self.domains}
        return _epoch_stats(self.domains, group_sizes, self._image_counts, self._refill_counts)

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch) #uses seed and epoch as the random seed. for every epoch the collection is different, but every run is ultimately consistent
        self.epoch += 1

        self._image_counts = {d: 0 for d in self.domains}
        self._refill_counts = {d: 0 for d in self.domains}

        pools = {} #shuffled dictionary of entries per domain
        cursors = {} #pointer for each domain in the pool (index)

        #refills the indexes for one domain
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
    """caps how many domains show up per batch at domains_per_batch (2 by default)
    instead of letting batch_size decide

    this is closer to the source/target pair setup in sun & saenko's deep coral
    paper, a couple of same-domain groups per batch, each big enough that the
    covariance actually means something, samples-per-domain is
    batch_size // domains_per_batch, so unlike DomainDiverseBatchSampler it
    grows with batch_size instead of being stuck at 1, the pairs get picked at
    random every batch so training eventually sees all the combinations rather
    than one fixed pair

    small domains still get oversampled here, same as the other one, you can't
    really avoid that with domain balancing on something as lopsided as gwhd
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
        """per domain oversampling stats for whatever epoch just finished"""
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

        #split batch_size evenly over the chosen domains, leftovers go to the
        #first groups when it doesn't divide cleanly
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
    """no domain control whatsoever, batches are just random shuffles of the
    whole training set

    same thing a plain shuffle=True dataloader does, which is what the non-dg
    baseline and the original grl script (train_GWHD_dgfrcnn.py) both use, what
    lands in a batch is pure luck, so big domains like ethz_1 (747 images) turn
    up constantly and tiny ones like arvalis_8 (20 images) almost never, and
    samples-per-domain just scales with batch_size, this is the closest i get to
    not pinning domains to batch size

    only safeguard is two domains minimum per batch, because FRCNNCoralLosses
    zeroes out the coral loss on a single-domain batch and i'd rather not throw
    away the signal for that step, with ensure_min_domains on (the default) one
    image in an all-one-domain batch gets swapped for a random one from
    somewhere else, turn it off if you want the baseline's batching exactly
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
        """per domain draw stats for whatever epoch just finished

        no per-domain pools here unlike the other two samplers, so
        times_recycled is just the one shared full-dataset wraparound count and
        comes out the same for every domain
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
                #reshuffle and start over, basically a fresh dataloader epoch,
                #so we never run dry halfway through a batch
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
