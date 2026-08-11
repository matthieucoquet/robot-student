import torch
from genesis.utils.geom import inv_quat, transform_quat_by_quat
from tensordict import TensorClass

from robot_student.util.geometry import quat_to_rotation_vector


def _angular_displacement(from_rotation: torch.Tensor, to_rotation: torch.Tensor) -> torch.Tensor:
    relative_rotation = transform_quat_by_quat(inv_quat(from_rotation), to_rotation)
    return quat_to_rotation_vector(relative_rotation)


class Motion(TensorClass["autocast"]):
    frequency: int

    root_position: torch.Tensor
    root_rotation: torch.Tensor
    joint_dof_positions: torch.Tensor

    root_velocity: torch.Tensor | None = None
    root_angular_velocity: torch.Tensor | None = None

    joint_dof_velocities: torch.Tensor | None = None
    # joint_rotation: torch.Tensor | None = None

    # Computed later by FK
    link_positions: torch.Tensor | None = None
    link_rotations: torch.Tensor | None = None
    # link_velocities: torch.Tensor | None = None
    # link_angular_velocity: torch.Tensor | None = None

    @property
    def frame_count(self) -> int:
        return len(self)

    @torch.no_grad()
    def compute_velocities(self) -> None:
        if self.frame_count < 2:
            raise ValueError(f"At least two frames are required to compute velocities, got {self.frame_count}")

        self.root_velocity = torch.zeros_like(self.root_position)
        self.root_angular_velocity = torch.zeros_like(self.root_position)
        self.joint_dof_velocities = torch.zeros_like(self.joint_dof_positions)

        self.root_velocity[1:-1].copy_((self.root_position[2:] - self.root_position[:-2]) * (0.5 * self.frequency))
        self.root_velocity[0].copy_((self.root_position[1] - self.root_position[0]) * self.frequency)
        self.root_velocity[-1].copy_((self.root_position[-1] - self.root_position[-2]) * self.frequency)

        self.root_angular_velocity[1:-1].copy_(
            _angular_displacement(self.root_rotation[:-2], self.root_rotation[2:]) * (0.5 * self.frequency)
        )
        self.root_angular_velocity[0].copy_(_angular_displacement(self.root_rotation[0], self.root_rotation[1]) * self.frequency)
        self.root_angular_velocity[-1].copy_(_angular_displacement(self.root_rotation[-2], self.root_rotation[-1]) * self.frequency)

        self.joint_dof_velocities[1:-1].copy_((self.joint_dof_positions[2:] - self.joint_dof_positions[:-2]) * (0.5 * self.frequency))
        self.joint_dof_velocities[0].copy_((self.joint_dof_positions[1] - self.joint_dof_positions[0]) * self.frequency)
        self.joint_dof_velocities[-1].copy_((self.joint_dof_positions[-1] - self.joint_dof_positions[-2]) * self.frequency)
