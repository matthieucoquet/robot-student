import torch
from genesis.engine.entities import KinematicEntity


class KinematicRobot:
    def __init__(self, entity: KinematicEntity) -> None:
        self._entity: KinematicEntity = entity
        self.n_qs = self._entity.n_qs
        self.n_dofs = self._entity.n_dofs
        self.n_root_dofs = self._entity.links[0].n_dofs
        self.n_joint_dofs = self.n_dofs - self.n_root_dofs

    def get_joint_dof_positions(self, environment_indices: torch.Tensor | None = None) -> torch.Tensor:
        positions = self._entity.get_dofs_position(envs_idx=environment_indices)
        return positions[..., self.n_root_dofs :]

    def get_joint_dof_velocities(self, environment_indices: torch.Tensor | None = None) -> torch.Tensor:
        dofs_velocity = self._entity.get_dofs_velocity(envs_idx=environment_indices)
        return dofs_velocity[..., self.n_root_dofs :]

    def get_links_position(self, environment_indices: torch.Tensor | None = None, relative: bool = False) -> torch.Tensor:
        return self._entity.get_links_pos(envs_idx=environment_indices, relative=relative)

    def get_links_rotation(self, environment_indices: torch.Tensor | None = None, relative: bool = False) -> torch.Tensor:
        return self._entity.get_links_quat(envs_idx=environment_indices, relative=relative)

    # def get_links_velocity(self, environment_indices: torch.Tensor | None = None) -> torch.Tensor:
    #     return self._entity.get_links_vel(envs_idx=environment_indices)

    # def get_links_angular_velocity(self, environment_indices: torch.Tensor | None = None) -> torch.Tensor:
    #     return self._entity.get_links_ang(envs_idx=environment_indices)

    def set_generalized_positions(
        self,
        generalized_positions: torch.Tensor,
        zero_velocity: bool = False,
        environment_indices: torch.Tensor | None = None,
    ) -> None:
        self._entity.set_qpos(generalized_positions, envs_idx=environment_indices, zero_velocity=zero_velocity)

    def set_generalized_state(
        self,
        generalized_positions: torch.Tensor,
        velocities: torch.Tensor | None = None,
        environment_indices: torch.Tensor | None = None,
    ) -> None:
        if velocities is None:
            self._entity.set_qpos(generalized_positions, envs_idx=environment_indices, zero_velocity=True)
            return

        self._entity.set_dofs_velocity(velocity=velocities, envs_idx=environment_indices, skip_forward=True)
        self._entity.set_qpos(generalized_positions, envs_idx=environment_indices, zero_velocity=False)
