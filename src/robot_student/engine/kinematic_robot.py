from collections.abc import Sequence

import torch
from genesis.engine.entities import KinematicEntity
from genesis.utils.geom import inv_transform_by_quat
from tensordict import TensorClass


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

    def copy_environments_(self, environment_indices: torch.Tensor, source: "RobotState") -> None:
        GeneralizedRobotState.copy_environments_(self, environment_indices, source)
        self.world_link_positions.index_copy_(0, environment_indices, source.world_link_positions)


class KinematicRobot:
    def __init__(self, entity: KinematicEntity) -> None:
        self._entity: KinematicEntity = entity
        self.n_links = self._entity.n_links
        self.n_qs = self._entity.n_qs
        self.n_dofs = self._entity.n_dofs
        self.n_root_qs = self._entity.links[0].n_qs
        self.n_root_dofs = self._entity.links[0].n_dofs
        self.n_joint_dofs = self.n_dofs - self.n_root_dofs

        self.joint_dof_names = self._get_joint_dof_names()

    def _get_joint_dof_names(self) -> tuple[str, ...]:
        joint_dof_names = tuple(
            joint.name if joint.n_dofs == 1 else f"{joint.name}[{offset}]"
            for joint in self._entity.joints
            for offset, dof_index in enumerate(joint.dofs_idx_local)
            if dof_index >= self.n_root_dofs
        )
        if len(joint_dof_names) != self.n_joint_dofs:
            raise RuntimeError(f"Expected {self.n_joint_dofs} named joint DOFs, found {len(joint_dof_names)}")
        return joint_dof_names

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
        generalized_positions = self._entity.get_qpos(envs_idx=environment_indices)
        generalized_velocities = self._entity.get_dofs_velocity(envs_idx=environment_indices)
        return RobotState(
            root_position=generalized_positions[..., :3],
            root_rotation=generalized_positions[..., 3 : self.n_root_qs],
            joint_dof_positions=generalized_positions[..., self.n_root_qs :],
            root_velocity=generalized_velocities[..., :3],
            root_angular_velocity=self._entity.get_ang(envs_idx=environment_indices),
            joint_dof_velocities=generalized_velocities[..., self.n_root_dofs :],
            world_link_positions=self._entity.get_links_pos(envs_idx=environment_indices, relative=False),
            batch_size=generalized_positions.shape[:-1],
        )

    def get_links_position(
        self,
        environment_indices: torch.Tensor | None = None,
        relative: bool = False,
    ) -> torch.Tensor:
        return self._entity.get_links_pos(
            envs_idx=environment_indices,
            relative=relative,
        )

    def get_links_rotation(self, environment_indices: torch.Tensor | None = None, relative: bool = False) -> torch.Tensor:
        return self._entity.get_links_quat(envs_idx=environment_indices, relative=relative)

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

    def set_state(self, state: GeneralizedRobotState, environment_indices: torch.Tensor | None = None) -> RobotState:
        generalized_positions = torch.cat(
            (state.root_position, state.root_rotation, state.joint_dof_positions),
            dim=-1,
        )
        local_root_angular_velocity = inv_transform_by_quat(state.root_angular_velocity, state.root_rotation)
        generalized_velocities = torch.cat(
            (state.root_velocity, local_root_angular_velocity, state.joint_dof_velocities),
            dim=-1,
        )
        self._set_generalized_state(
            generalized_positions,
            generalized_velocities,
            environment_indices=environment_indices,
        )
        return RobotState(
            root_position=state.root_position,
            root_rotation=state.root_rotation,
            joint_dof_positions=state.joint_dof_positions,
            root_velocity=state.root_velocity,
            root_angular_velocity=state.root_angular_velocity,
            joint_dof_velocities=state.joint_dof_velocities,
            world_link_positions=self.get_links_position(environment_indices=environment_indices),
            batch_size=state.batch_size,
        )
