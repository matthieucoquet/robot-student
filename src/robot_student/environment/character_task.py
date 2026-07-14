from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import NamedTuple

import torch


class CharacterTaskStep(NamedTuple):
    reward: torch.Tensor
    terminal: torch.Tensor
    transition_metrics: Mapping[str, torch.Tensor]


class CharacterTask(ABC):
    @abstractmethod
    def step(
        self,
        root_position: torch.Tensor,
        root_velocity: torch.Tensor,
        normalized_control_forces: torch.Tensor,
    ) -> CharacterTaskStep:
        """Compute reward, termination, and diagnostic metrics for one simulation step."""
