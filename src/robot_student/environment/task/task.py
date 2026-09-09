from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, NamedTuple

import torch

from robot_student.engine.kinematic_robot import RobotState


class TaskStep(NamedTuple):
    reward: torch.Tensor
    terminal: torch.Tensor
    transition_metrics: Mapping[str, torch.Tensor]


class Task(ABC):
    @abstractmethod
    def step(self, state: RobotState, **kwargs: Any) -> TaskStep:
        """Compute one task step from the robot state and task-specific inputs."""
