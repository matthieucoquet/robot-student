from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, NamedTuple

import torch

from robot_student.engine.kinematic_robot import RobotState
from robot_student.engine.robot import Robot
from robot_student.environment.schema import TensorSchema

if TYPE_CHECKING:
    from robot_student.engine.genesis_engine import GenesisEngine


class TaskFeedback(NamedTuple):
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
        key_link_indices: torch.Tensor,
        simulation_steps_per_control_step: int,
        global_observation: bool,
    ) -> None:
        """Initialize after scene construction with the robot, read-only key link indices, and timing and observation settings."""

    @abstractmethod
    def reset(self, environment_indices: torch.Tensor) -> None:
        """Reset the task, change the scene if necessary. for this task"""

    def step(self, is_control_step: bool) -> None:
        """Advance task state before a simulation step; is_control_step marks the first step of a control interval."""
        return None

    @abstractmethod
    def compute_feedback(self, state: RobotState, **kwargs: Any) -> TaskFeedback:
        """Compute reward, termination, and metrics from the robot state and task-specific inputs."""

    @abstractmethod
    def observation(self, robot_state: RobotState, *, previous_action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return task observations. previous_action is a read-only buffer, zero on reset; clone it if returning it directly."""
