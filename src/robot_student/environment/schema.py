from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TensorSchema:
    shape: tuple[int, ...]
    data_type: torch.dtype
    bounds: tuple[torch.Tensor, torch.Tensor] | None = None


@dataclass(frozen=True)
class EnvironmentSchema:
    observations: dict[str, TensorSchema]
    actions: dict[str, TensorSchema]
