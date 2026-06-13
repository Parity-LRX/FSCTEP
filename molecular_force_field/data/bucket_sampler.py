"""Bucketed batch sampler for make_fx-compiled training.

Every batch it yields contains only samples from a single size-bucket (see
``H5Dataset(makefx_buckets=...)``), so each batch has ONE fixed graph shape and the
make_fx-compiled force step is reused per bucket (one compile each) instead of recompiling
per unique molecule size — while padding far less than padding everything to the global max.

DDP-aware: each rank gets a disjoint, equal-count shard of the batch list (remainder dropped so
every rank does the same number of steps -> no collective hang). Reshuffle each epoch via
``set_epoch`` (same contract as ``DistributedSampler``).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterator, List, Sequence

import torch
from torch.utils.data import Sampler


class BucketBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        sample_bucket: Sequence[int],
        batch_size: int,
        *,
        shuffle: bool = True,
        drop_last: bool = False,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
    ) -> None:
        if sample_bucket is None:
            raise ValueError("BucketBatchSampler needs dataset.sample_bucket (build the dataset "
                             "with makefx_buckets=...).")
        self.sample_bucket = list(sample_bucket)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.num_replicas = max(int(num_replicas), 1)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self._by_bucket: dict[int, List[int]] = defaultdict(list)
        for i, b in enumerate(self.sample_bucket):
            self._by_bucket[int(b)].append(i)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _build_batches(self) -> List[List[int]]:
        gen = torch.Generator()
        gen.manual_seed(self.seed + self.epoch)
        batches: List[List[int]] = []
        for b in sorted(self._by_bucket):
            idxs = self._by_bucket[b]
            if self.shuffle:
                order = torch.randperm(len(idxs), generator=gen).tolist()
                idxs = [idxs[i] for i in order]
            for s in range(0, len(idxs), self.batch_size):
                batch = idxs[s:s + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch)
        if self.shuffle:
            order = torch.randperm(len(batches), generator=gen).tolist()
            batches = [batches[i] for i in order]
        # DDP: keep only a whole multiple of num_replicas so every rank gets the same count.
        if self.num_replicas > 1:
            n = (len(batches) // self.num_replicas) * self.num_replicas
            batches = batches[self.rank:n:self.num_replicas]
        return batches

    def __iter__(self) -> Iterator[List[int]]:
        yield from self._build_batches()

    def __len__(self) -> int:
        return len(self._build_batches())
