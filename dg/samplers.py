from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterator, List

from torch.utils.data import Sampler


class DomainDiverseBatchSampler(Sampler[List[int]]):
    """
    Domain-balanced batch sampler that draws distinct domains within each batch.

    Designed for CORAL training where a cross-domain loss is zero unless at
    least two domains occur in the minibatch. Smaller domains are naturally
    oversampled because exhausted domain buckets are reshuffled and reused.
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

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch) #uses seed and epoch as the random seed. for every epoch the collection is different, but every run is ultimately consistent
        self.epoch += 1

        pools = {} #shuffled dictionary of entries per domain
        cursors = {} #pointer for each domain in the pool (index)

        #refills the indexes for one domain. 
        def refill(domain: int):
            values = list(self.groups[domain])
            rng.shuffle(values)
            pools[domain] = values
            cursors[domain] = 0

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

            yield batch
