from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, NamedTuple

import torch

from robot_student.engine.kinematic_robot import RobotState


class CharacterTaskStep(NamedTuple):
    reward: torch.Tensor
    terminal: torch.Tensor
    transition_metrics: Mapping[str, torch.Tensor]


class CharacterTask(ABC):
    @abstractmethod
    def step(self, state: RobotState, **kwargs: Any) -> CharacterTaskStep:
        """Compute one task step from the robot state and task-specific inputs."""
