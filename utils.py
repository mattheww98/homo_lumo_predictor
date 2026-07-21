from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Subset


@dataclass
class Standardizer:
    mean: torch.Tensor | float
    std: torch.Tensor | float

    def __post_init__(self):
        self.mean = torch.as_tensor(self.mean, dtype=torch.float32)
        self.std = torch.as_tensor(self.std, dtype=torch.float32)
        self.std = torch.where(self.std.abs() < 1e-12, torch.ones_like(self.std), self.std)

    @classmethod
    def fit(cls, values: torch.Tensor, dim: int | tuple[int, ...] | None = None):
        values = torch.as_tensor(values, dtype=torch.float32)
        return cls(values.mean(dim=dim), values.std(dim=dim, unbiased=False))

    def _stats_for(self, values: torch.Tensor):
        return (
            self.mean.to(device=values.device, dtype=values.dtype),
            self.std.to(device=values.device, dtype=values.dtype),
        )

    def scale(self, values: torch.Tensor) -> torch.Tensor:
        mean, std = self._stats_for(values)
        return (values - mean) / std

    def unscale(self, values: torch.Tensor) -> torch.Tensor:
        mean, std = self._stats_for(values)
        return values * std + mean

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"mean": self.mean.cpu(), "std": self.std.cpu()}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, torch.Tensor]):
        return cls(mean=state["mean"], std=state["std"])


def load_config(path: str | Path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_dataset(
    dataset: Sequence[Any],
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int,
):
    requested = train_size + val_size + test_size
    if requested > len(dataset):
        raise ValueError(f"Requested {requested} samples, but the dataset has {len(dataset)}")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:requested].tolist()
    train_end = train_size
    val_end = train_end + val_size
    return (
        Subset(dataset, indices[:train_end]),
        Subset(dataset, indices[train_end:val_end]),
        Subset(dataset, indices[val_end:]),
    )


def fit_target_standardizer(dataset: Sequence[Any], target_index: int):
    targets = torch.stack([torch.as_tensor(data.y).reshape(-1)[target_index] for data in dataset])
    return Standardizer.fit(targets)


def fit_node_standardizer(
    dataset: Sequence[Any],
    feature_indices: Sequence[int],
    continuous_feature_indices: Sequence[int],
):
    feature_indices = list(feature_indices)
    continuous = set(continuous_feature_indices)
    nodes = torch.cat([torch.as_tensor(data.x)[:, feature_indices] for data in dataset], dim=0).float()
    mean = torch.zeros(len(feature_indices), dtype=nodes.dtype)
    std = torch.ones(len(feature_indices), dtype=nodes.dtype)
    positions = [i for i, original_index in enumerate(feature_indices) if original_index in continuous]
    if positions:
        fitted = Standardizer.fit(nodes[:, positions], dim=0)
        mean[positions] = fitted.mean
        std[positions] = fitted.std
    return Standardizer(mean, std)


def resolve_path(path: str | Path, relative_to: str | Path):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = Path(relative_to) / path
    return path.resolve()


def count_parameters(model: torch.nn.Module):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
