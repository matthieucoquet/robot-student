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


class GeneralizedRobotState(TensorClass["autocast"]):
    root_position: torch.Tensor
    root_rotation: torch.Tensor
    joint_dof_positions: torch.Tensor
    root_velocity: torch.Tensor  # World coordinates
    root_angular_velocity: torch.Tensor  # World coordinates
    joint_dof_velocities: torch.Tensor

    def copy_environments_(self, environment_indices: torch.Tensor, source: "GeneralizedRobotState") -> None:
        self.root_position.index_copy_(0, environment_indices, source.root_position)
        self.root_rotation.index_copy_(0, environment_indices, source.root_rotation)
        self.joint_dof_positions.index_copy_(0, environment_indices, source.joint_dof_positions)
        self.root_velocity.index_copy_(0, environment_indices, source.root_velocity)
        self.root_angular_velocity.index_copy_(0, environment_indices, source.root_angular_velocity)
        self.joint_dof_velocities.index_copy_(0, environment_indices, source.joint_dof_velocities)


class RobotState(GeneralizedRobotState):
    world_link_positions: torch.Tensor
    world_link_rotations: torch.Tensor  # wxyz quaternions
    world_link_linear_velocities: torch.Tensor
    world_link_angular_velocities: torch.Tensor

    def copy_environments_(self, environment_indices: torch.Tensor, source: "RobotState") -> None:
        GeneralizedRobotState.copy_environments_(self, environment_indices, source)
        self.world_link_positions.index_copy_(0, environment_indices, source.world_link_positions)
        self.world_link_rotations.index_copy_(0, environment_indices, source.world_link_rotations)
        self.world_link_linear_velocities.index_copy_(0, environment_indices, source.world_link_linear_velocities)
        self.world_link_angular_velocities.index_copy_(0, environment_indices, source.world_link_angular_velocities)
