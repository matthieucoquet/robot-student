from pathlib import Path
from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict, TensorDictBase

from robot_student.control_mode import ControlMode
from robot_student.environment.environment import Environment
from robot_student.environment.schema import EnvironmentSchema, TensorSchema

if TYPE_CHECKING:
    from robot_student.engine.genesis_engine import GenesisEngine


class CharacterEnvironment(Environment):
    def __init__(self, engine: "GenesisEngine", xml_path: Path, environment_count: int, control_mode: ControlMode) -> None:
        self._engine = engine
        self._engine.add_ground_plane()
        self._character = engine.add_character(xml_path, control_mode=control_mode)
        self._engine.build_scene(environment_count=environment_count, env_spacing=(1.0, 1.0))

        self._schema = self._compute_schema()

    @property
    def schema(self) -> EnvironmentSchema:
        return self._schema

    def reset(self) -> TensorDictBase:
        self._engine.reset()

        return self._get_observation()

    def step(self, action: TensorDictBase) -> TensorDictBase:
        self._engine.step(action)

        return self._get_observation()

    def _compute_schema(self) -> EnvironmentSchema:
        observation_type = torch.float32

        return EnvironmentSchema(
            observations={
                "proprioception": TensorSchema(
                    shape=(self._character.n_qs + self._character.n_dofs,),
                    data_type=observation_type,
                )
            },
            actions={"control": self._character.get_action_schema()},
        )

    def _get_observation(self) -> TensorDictBase:
        proprioception = torch.cat((self._character.get_root_position(), self._character.get_joint_positions()), dim=-1)

        return TensorDict({"proprioception": proprioception}, batch_size=proprioception.shape[:-1], device=proprioception.device)
