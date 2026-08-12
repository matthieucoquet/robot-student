import torch
from tensordict import TensorClass


class Motion(TensorClass["autocast"]):
    frequency: int

    root_position: torch.Tensor
    root_rotation: torch.Tensor
    joint_dof: torch.Tensor

    # Computed later by FK
    link_positions: torch.Tensor | None = None
    link_rotations: torch.Tensor | None = None

    @property
    def frame_count(self) -> int:
        return len(self)

    def get_generalized_positions(self, index) -> torch.Tensor:
        return torch.cat(
            [
                self.root_position[index],
                self.root_rotation[index],
                self.joint_dof[index],
            ],
            dim=-1,
        )
