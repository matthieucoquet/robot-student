from abc import ABC, abstractmethod

import torch
from tensordict import TensorDictBase


class CharacterTask(ABC):
    @abstractmethod
    def step(
        self,
        generalized_positions: torch.Tensor,
        generalized_velocities: torch.Tensor,
        action: TensorDictBase,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute reward, terminal, and truncation tensors for one simulation step."""
