from collections.abc import Sequence
from dataclasses import dataclass

import torch
from genesis.engine.entities import KinematicEntity


@dataclass(frozen=True, kw_only=True, slots=True)
class RobotState:
    root_position: torch.Tensor
    root_rotation: torch.Tensor
    joint_dof_positions: torch.Tensor
    root_velocity: torch.Tensor
    root_angular_velocity: torch.Tensor
    joint_dof_velocities: torch.Tensor
    link_positions: torch.Tensor
    # link_rotations: torch.Tensor | None

    def copy_environments_(self, environment_indices: torch.Tensor, source: "RobotState") -> None:
        self.root_position.index_copy_(0, environment_indices, source.root_position)
        self.root_rotation.index_copy_(0, environment_indices, source.root_rotation)
        self.joint_dof_positions.index_copy_(0, environment_indices, source.joint_dof_positions)
        self.root_velocity.index_copy_(0, environment_indices, source.root_velocity)
        self.root_angular_velocity.index_copy_(0, environment_indices, source.root_angular_velocity)
        self.joint_dof_velocities.index_copy_(0, environment_indices, source.joint_dof_velocities)
        self.link_positions.index_copy_(0, environment_indices, source.link_positions)


class KinematicRobot:
    def __init__(self, entity: KinematicEntity) -> None:
        self._entity: KinematicEntity = entity
        self.n_qs = self._entity.n_qs
        self.n_dofs = self._entity.n_dofs
        self.n_root_dofs = self._entity.links[0].n_dofs
        self.n_joint_dofs = self.n_dofs - self.n_root_dofs

    def get_link_indices(self, link_names: Sequence[str]) -> tuple[int, ...]:
        link_indices_by_name = {link.name: link.idx_local for link in self._entity.links}
        link_indices = [link_indices_by_name[link] for link in link_names]
        return link_indices

    def get_joint_dof_positions(self, environment_indices: torch.Tensor | None = None) -> torch.Tensor:
        positions = self._entity.get_dofs_position(envs_idx=environment_indices)
        return positions[..., self.n_root_dofs :]

    def get_joint_dof_velocities(self, environment_indices: torch.Tensor | None = None) -> torch.Tensor:
        dofs_velocity = self._entity.get_dofs_velocity(envs_idx=environment_indices)
        return dofs_velocity[..., self.n_root_dofs :]

    def get_state(self, environment_indices: torch.Tensor | None = None) -> RobotState:
        return RobotState(
            root_position=self._entity.get_pos(envs_idx=environment_indices, relative=False),
            root_rotation=self._entity.get_quat(envs_idx=environment_indices, relative=False),
            joint_dof_positions=self.get_joint_dof_positions(environment_indices),
            root_velocity=self._entity.get_vel(envs_idx=environment_indices),
            root_angular_velocity=self._entity.get_ang(envs_idx=environment_indices),
            joint_dof_velocities=self.get_joint_dof_velocities(environment_indices),
            link_positions=self._entity.get_links_pos(envs_idx=environment_indices, relative=False),
        )

    # def get_links_position(
    #     self,
    #     link_indices: Sequence[int] | torch.Tensor | None = None,
    #     environment_indices: torch.Tensor | None = None,
    #     relative: bool = False,
    # ) -> torch.Tensor:
    #     return self._entity.get_links_pos(
    #         links_idx_local=link_indices,
    #         envs_idx=environment_indices,
    #         relative=relative,
    #     )

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

    def _set_generalized_state(
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

    def set_state(self, state: RobotState, environment_indices: torch.Tensor | None = None) -> None:
        generalized_positions = torch.cat(
            (state.root_position, state.root_rotation, state.joint_dof_positions),
            dim=-1,
        )
        generalized_velocities = torch.cat(
            (state.root_velocity, state.root_angular_velocity, state.joint_dof_velocities),
            dim=-1,
        )
        self._set_generalized_state(
            generalized_positions,
            generalized_velocities,
            environment_indices=environment_indices,
        )
