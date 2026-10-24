from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, NamedTuple

import torch

from robot_student.engine.robot import Robot
from robot_student.environment.schema import TensorSchema

if TYPE_CHECKING:
    from robot_student.engine.genesis_engine import GenesisEngine


class TaskEvaluation(NamedTuple):
    reward: torch.Tensor
    terminal: torch.Tensor
    transition_metrics: Mapping[str, torch.Tensor]


class Task(ABC):
    def setup_scene(self, engine: "GenesisEngine") -> None:
        """Add task entities once, after the controlled robot is added and before the environment builds the scene."""
        return None

    @abstractmethod
    def get_schema(self) -> dict[str, TensorSchema]:
        """Return schemas for task observations, excluding the environment batch dimensions."""

    @abstractmethod
    def initialize(
        self,
        *,
        robot: Robot,
        simulation_steps_per_control_step: int,
    ) -> None:
        """Initialize after scene construction with the robot."""

    @abstractmethod
    def reset(self, environment_indices: torch.Tensor | None = None) -> None:
        """Reset selected environments and seed their observations and histories without advancing sensor time.

        Install any task-specific reset pose before capturing the initial readings. None selects all environments.
        """

    def step(self, is_control_step: bool) -> None:
        """Advance task state before a simulation step; is_control_step marks the first step of a control interval."""
        return None

    @abstractmethod
    def measure_state(self) -> None:
        """Capture the full batch once after each control interval, before evaluation and observation construction.

        This is the only control-step hook that samples measurements and advances sensor histories.
        """

    @abstractmethod
    def evaluate_step(self, *, current_action: torch.Tensor, previous_action: torch.Tensor) -> TaskEvaluation:
        """Compute reward, termination, and metrics from cached observations without sampling or advancing sensor histories."""

    @abstractmethod
    def observation(self, *, previous_action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Build features matching get_schema from cached observations without sampling or advancing sensor histories."""
