import torch
from genesis.engine.entities import KinematicEntity


class BaseRobot:
    def __init__(self, character: KinematicEntity) -> None:
        self._character: KinematicEntity = character

    def get_generalized_positions(self, environment_indices: torch.Tensor | None = None) -> torch.Tensor:
        return self._character.get_qpos(envs_idx=environment_indices)

    def get_generalized_velocities(self, environment_indices: torch.Tensor | None = None) -> torch.Tensor:
        return self._character.get_dofs_velocity(envs_idx=environment_indices)

    def get_links_position(self, environment_indices: torch.Tensor | None = None, relative: bool = False) -> torch.Tensor:
        return self._character.get_links_pos(envs_idx=environment_indices, relative=relative)

    def get_links_rotation(self, environment_indices: torch.Tensor | None = None, relative: bool = False) -> torch.Tensor:
        return self._character.get_links_quat(envs_idx=environment_indices, relative=relative)

    # def get_links_velocity(self, environment_indices: torch.Tensor | None = None) -> torch.Tensor:
    #     return self._character.get_links_vel(envs_idx=environment_indices)

    # def get_links_angular_velocity(self, environment_indices: torch.Tensor | None = None) -> torch.Tensor:
    #     return self._character.get_links_ang(envs_idx=environment_indices)

    def set_generalized_positions(
        self,
        generalized_positions: torch.Tensor,
        zero_velocity: bool = False,
        environment_indices: torch.Tensor | None = None,
    ) -> None:
        self._character.set_qpos(generalized_positions, envs_idx=environment_indices, zero_velocity=zero_velocity)

    def set_generalized_state(
        self,
        generalized_positions: torch.Tensor,
        velocities: torch.Tensor | None = None,
        environment_indices: torch.Tensor | None = None,
    ) -> None:
        if velocities is None:
            self._character.set_qpos(generalized_positions, envs_idx=environment_indices, zero_velocity=True)
            return

        self._character.set_dofs_velocity(velocity=velocities, envs_idx=environment_indices, skip_forward=True)
        self._character.set_qpos(generalized_positions, envs_idx=environment_indices, zero_velocity=False)
