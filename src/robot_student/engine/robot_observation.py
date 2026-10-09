from dataclasses import dataclass

import torch
from tensordict import TensorClass


@dataclass(frozen=True, kw_only=True, slots=True)
class UniformNoise:
    half_width: float


@dataclass(frozen=True, kw_only=True, slots=True)
class NoiseConfiguration:
    root_position: UniformNoise | None = None
    root_rotation: UniformNoise | None = None
    joint_dof_positions: UniformNoise | None = None
    root_velocity: UniformNoise | None = None
    root_angular_velocity: UniformNoise | None = None
    joint_dof_velocities: UniformNoise | None = None

    world_link_positions: UniformNoise | None = None
    world_link_rotations: UniformNoise | None = None
    world_link_linear_velocities: UniformNoise | None = None
    world_link_angular_velocities: UniformNoise | None = None


class RobotObservation(TensorClass["autocast"]):
    """Read-only robot measurements and estimates, potentially captured at different times.

    Fields mirror RobotState for task feature construction, without promising physical consistency.
    Unchanged fields may share storage with clean state. Rotations are wxyz quaternions;
    root and link velocities are expressed in world coordinates.
    """

    root_position: torch.Tensor
    root_rotation: torch.Tensor
    joint_dof_positions: torch.Tensor
    root_velocity: torch.Tensor
    root_angular_velocity: torch.Tensor
    joint_dof_velocities: torch.Tensor
    world_link_positions: torch.Tensor
    world_link_rotations: torch.Tensor
    world_link_linear_velocities: torch.Tensor
    world_link_angular_velocities: torch.Tensor

    def copy_environments_(self, environment_indices: torch.Tensor, source: "RobotObservation") -> None:
        """Replace selected cached measurements without changing other environments."""
        for field_name, value in self.items():
            value.index_copy_(0, environment_indices, getattr(source, field_name))
