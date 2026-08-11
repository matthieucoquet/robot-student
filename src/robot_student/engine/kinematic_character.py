import genesis as gs
import torch
from genesis.engine.entities import KinematicEntity


class KinematicCharacter:
    def __init__(self, character: KinematicEntity) -> None:
        self._character: KinematicEntity = character

    def get_generalized_positions(self, environment_indices: torch.Tensor | None = None) -> torch.Tensor:
        return self._character.get_qpos(envs_idx=environment_indices)

    def get_generalized_velocities(self, environment_indices: torch.Tensor | None = None) -> torch.Tensor:
        return self._character.get_dofs_velocity(envs_idx=environment_indices)

    def set_generalized_positions(self, generalized_positions: torch.Tensor, zero_velocity: bool = False) -> None:
        self._character.set_qpos(generalized_positions, zero_velocity=zero_velocity)
