from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch.utils.data import Dataset, Sampler

from .constants import PAD_BYTE, ROLE_TO_ID, TYPE_TO_ID


@dataclass(frozen=True)
class Sample:
    sample_id: str
    protocol: str
    direction: str
    byte_values: tuple[int, ...]
    boundaries: tuple[int, ...]
    role_ids: tuple[int, ...]
    type_ids: tuple[int, ...]

    @property
    def length(self) -> int:
        return len(self.byte_values)


def _validate_sample(sample: Sample) -> None:
    n = sample.length
    if not n:
        raise ValueError(f"{sample.sample_id}: empty message")
    if any(not 0 <= value <= 255 for value in sample.byte_values):
        raise ValueError(f"{sample.sample_id}: invalid byte value")
    if any(not 0 < cut < n for cut in sample.boundaries):
        raise ValueError(f"{sample.sample_id}: boundaries must be internal cut positions")
    if tuple(sorted(set(sample.boundaries))) != sample.boundaries:
        raise ValueError(f"{sample.sample_id}: boundaries must be sorted and unique")
    if len(sample.role_ids) != n or len(sample.type_ids) != n:
        raise ValueError(f"{sample.sample_id}: semantic labels do not tile message")


def sample_from_dict(row: dict) -> Sample:
    sample = Sample(
        sample_id=str(row["id"]),
        protocol=str(row["protocol"]),
        direction=str(row.get("direction", "?")),
        byte_values=tuple(int(x) for x in row["bytes"]),
        boundaries=tuple(int(x) for x in row["boundaries"]),
        role_ids=tuple(int(x) for x in row["role_ids"]),
        type_ids=tuple(int(x) for x in row["type_ids"]),
    )
    _validate_sample(sample)
    return sample


def load_jsonl(path: str | Path) -> list[Sample]:
    samples = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                samples.append(sample_from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not samples:
        raise ValueError(f"no samples in {path}")
    return samples


class ProtocolDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Sample],
        max_length: int | None = None,
        random_crop: bool = False,
        seed: int = 0,
    ) -> None:
        self.samples = list(samples)
        self.max_length = max_length
        self.random_crop = random_crop
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        start, end = 0, sample.length
        if self.max_length and sample.length > self.max_length:
            start = self.rng.randint(0, sample.length - self.max_length) if self.random_crop else 0
            end = start + self.max_length

        boundaries = [cut - start for cut in sample.boundaries if start < cut < end]
        n = end - start
        boundary_labels = torch.zeros(n, dtype=torch.float32)
        if boundaries:
            boundary_labels[torch.tensor(boundaries, dtype=torch.long)] = 1.0

        return {
            "id": sample.sample_id,
            "protocol": sample.protocol,
            "direction": sample.direction,
            "bytes": torch.tensor(sample.byte_values[start:end], dtype=torch.long),
            "boundary_labels": boundary_labels,
            "role_labels": torch.tensor(sample.role_ids[start:end], dtype=torch.long),
            "type_labels": torch.tensor(sample.type_ids[start:end], dtype=torch.long),
            "length": n,
            "original_length": sample.length,
            "crop_start": start,
        }


class FixedRatioBatchSampler(Sampler[list[int]]):
    """Build deterministic full batches with a fixed auxiliary-data ratio.

    Base indices occupy ``[0, base_size)`` and long-message indices follow them.
    Every base example is seen at least once per epoch; the final incomplete base
    block and the (usually smaller) long set are sampled with replacement.  This
    keeps the per-batch composition identical across model architectures.
    """

    def __init__(
        self,
        base_size: int,
        long_size: int,
        batch_size: int,
        long_fraction: float,
        seed: int,
    ) -> None:
        if base_size <= 0 or long_size <= 0:
            raise ValueError("base and long datasets must both be non-empty")
        if batch_size < 2:
            raise ValueError("fixed-ratio batches require batch_size >= 2")
        if not 0.0 < long_fraction < 1.0:
            raise ValueError("long_fraction must be strictly between 0 and 1")
        long_per_batch = int(round(batch_size * long_fraction))
        self.long_per_batch = min(max(long_per_batch, 1), batch_size - 1)
        self.base_per_batch = batch_size - self.long_per_batch
        self.base_size = base_size
        self.long_size = long_size
        self.batch_size = batch_size
        self.long_fraction = long_fraction
        self.seed = seed
        self.epoch = 0
        self.num_batches = math.ceil(base_size / self.base_per_batch)

    @property
    def effective_long_fraction(self) -> float:
        return self.long_per_batch / self.batch_size

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1

        base_indices = list(range(self.base_size))
        rng.shuffle(base_indices)
        missing = self.num_batches * self.base_per_batch - len(base_indices)
        if missing:
            base_indices.extend(rng.choices(base_indices, k=missing))

        for batch_index in range(self.num_batches):
            start = batch_index * self.base_per_batch
            batch = base_indices[start : start + self.base_per_batch]
            batch.extend(
                self.base_size + rng.randrange(self.long_size)
                for _ in range(self.long_per_batch)
            )
            rng.shuffle(batch)
            yield batch


def collate_samples(rows: Sequence[dict]) -> dict:
    batch_size = len(rows)
    max_len = max(row["length"] for row in rows)
    byte_values = torch.full((batch_size, max_len), PAD_BYTE, dtype=torch.long)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    boundary = torch.zeros((batch_size, max_len), dtype=torch.float32)
    role = torch.full((batch_size, max_len), -100, dtype=torch.long)
    type_labels = torch.full((batch_size, max_len), -100, dtype=torch.long)

    for idx, row in enumerate(rows):
        n = row["length"]
        byte_values[idx, :n] = row["bytes"]
        mask[idx, :n] = True
        boundary[idx, :n] = row["boundary_labels"]
        role[idx, :n] = row["role_labels"]
        type_labels[idx, :n] = row["type_labels"]

    return {
        "ids": [row["id"] for row in rows],
        "protocols": [row["protocol"] for row in rows],
        "directions": [row["direction"] for row in rows],
        "bytes": byte_values,
        "mask": mask,
        "boundary_labels": boundary,
        "role_labels": role,
        "type_labels": type_labels,
        "lengths": torch.tensor([row["length"] for row in rows], dtype=torch.long),
        "original_lengths": torch.tensor(
            [row["original_length"] for row in rows], dtype=torch.long
        ),
        "crop_starts": torch.tensor([row["crop_start"] for row in rows], dtype=torch.long),
    }


def normalized_row(
    sample_id: str,
    protocol: str,
    direction: str,
    byte_values: bytes,
    fields: Iterable[dict],
) -> dict:
    fields = sorted(fields, key=lambda field: int(field["off"]))
    n = len(byte_values)
    cursor = 0
    role_ids = [-1] * n
    type_ids = [-1] * n
    boundaries = []
    for idx, field in enumerate(fields):
        off, width = int(field["off"]), int(field["len"])
        if off != cursor or width <= 0 or off + width > n:
            raise ValueError(f"non-contiguous field at index {idx}: off={off}, len={width}")
        if off:
            boundaries.append(off)
        role_id = ROLE_TO_ID[field["role"]]
        type_id = TYPE_TO_ID[field["type"]]
        role_ids[off : off + width] = [role_id] * width
        type_ids[off : off + width] = [type_id] * width
        cursor += width
    if cursor != n:
        raise ValueError(f"fields cover {cursor} of {n} bytes")
    return {
        "id": sample_id,
        "protocol": protocol,
        "direction": direction,
        "bytes": list(byte_values),
        "boundaries": boundaries,
        "role_ids": role_ids,
        "type_ids": type_ids,
    }
