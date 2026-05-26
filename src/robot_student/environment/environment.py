from abc import ABC, abstractmethod

import torch
from tensordict import TensorDictBase

from robot_student.environment.schema import EnvironmentSchema


class Environment(ABC):
    @property
    @abstractmethod
    def schema(self) -> EnvironmentSchema:
        """Return the observation and action schema."""

    @abstractmethod
    def reset(self) -> TensorDictBase:
        """Reset the environment state and return the initial observation."""

    @abstractmethod
    def reset_done(self, done: torch.Tensor) -> TensorDictBase:
        """Reset completed environments and return the next observation."""

    @abstractmethod
    def step(self, action: TensorDictBase) -> tuple[TensorDictBase, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance the environment by one simulation step and return the observation."""
