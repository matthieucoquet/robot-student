from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import NamedTuple

import torch
from tensordict import TensorDictBase


class CharacterTaskStep(NamedTuple):
    reward: torch.Tensor
    terminal: torch.Tensor
    transition_metrics: Mapping[str, torch.Tensor]


class CharacterTask(ABC):
    @abstractmethod
    def step(
        self,
        generalized_positions: torch.Tensor,
        generalized_velocities: torch.Tensor,
        control_forces: torch.Tensor,
        action: TensorDictBase,
    ) -> CharacterTaskStep:
        """Compute reward, termination, and diagnostic metrics for one simulation step."""
