import torch
from tensordict import TensorClass


class GeneralizedRobotState(TensorClass["autocast"]):
    """Physical root and joint coordinates and velocities at one nominal instant."""

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
    """Physical robot kinematics, including link quantities, at one nominal instant."""

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
