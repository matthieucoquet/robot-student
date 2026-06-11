from abc import ABC, abstractmethod

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
    def step(self, action: TensorDictBase) -> TensorDictBase:
        """Advance the environment by one simulation step and return the observation."""
