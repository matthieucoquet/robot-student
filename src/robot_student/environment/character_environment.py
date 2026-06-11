from pathlib import Path

import genesis as gs
import torch
from tensordict import TensorDict, TensorDictBase

from robot_student.environment.environment import Environment
from robot_student.environment.schema import EnvironmentSchema, TensorSchema


class CharacterEnvironment(Environment):
    def __init__(self, xml_path: Path, environment_count: int = 1, device: str = "cpu", show_viewer: bool = True) -> None:
        gs.init(backend=gs.cuda if device == "cuda" else gs.cpu)

        self._scene = gs.Scene(show_viewer=show_viewer)
        self._scene.add_entity(gs.morphs.Plane())
        self._character = self._scene.add_entity(gs.morphs.MJCF(file=str(xml_path)))

        self._scene.build(n_envs=environment_count, env_spacing=(1.0, 1.0))
        self._controlled_dof_indices = self._compute_controlled_dof_indices()
        self._schema = self._compute_schema()

    @property
    def schema(self) -> EnvironmentSchema:
        return self._schema

    def reset(self) -> TensorDictBase:
        self._scene.reset()

        return self._get_observation()

    def step(self, action: TensorDictBase) -> TensorDictBase:
        self._character.control_dofs_position(action["control"], self._controlled_dof_indices)
        self._scene.step()

        return self._get_observation()

    def _compute_schema(self) -> EnvironmentSchema:
        observation_type = self._character.get_qpos().dtype

        lower_bounds, upper_bounds = self._character.get_dofs_limit(self._controlled_dof_indices)

        return EnvironmentSchema(
            observations={
                "proprioception": TensorSchema(
                    shape=(self._character.n_qs + self._character.n_dofs,),
                    data_type=observation_type,
                )
            },
            actions={
                "control": TensorSchema(
                    shape=(len(self._controlled_dof_indices),),
                    data_type=observation_type,
                    bounds=(lower_bounds, upper_bounds),
                )
            },
        )

    def _compute_controlled_dof_indices(self) -> list[int]:
        return [
            degree_of_freedom_index
            for joint in self._character.joints
            if joint.n_dofs > 0 and bool(joint.dofs_act_gain.any())
            for degree_of_freedom_index in joint.dofs_idx_local
        ]

    def _get_observation(self) -> TensorDictBase:
        proprioception = torch.cat((self._character.get_qpos(), self._character.get_dofs_velocity()), dim=-1)

        return TensorDict({"proprioception": proprioception}, batch_size=proprioception.shape[:-1], device=proprioception.device)
