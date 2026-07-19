from pathlib import Path
from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict, TensorDictBase

from robot_student.engine.control_mode import ControlMode
from robot_student.environment.character_task import CharacterTask
from robot_student.environment.environment import Environment
from robot_student.environment.schema import EnvironmentSchema, TensorSchema

if TYPE_CHECKING:
    from robot_student.engine.genesis_engine import GenesisEngine


class CharacterEnvironment(Environment):
    def __init__(
        self,
        engine: "GenesisEngine",
        xml_path: Path,
        environment_count: int,
        control_mode: ControlMode,
        task: CharacterTask,
        device: torch.device,
        maximum_episode_steps: int = 1_000,
    ) -> None:
        self._engine = engine
        self._task = task
        self._engine.add_ground_plane()
        self._character = engine.add_character(xml_path, control_mode=control_mode)
        self._engine.build_scene(environment_count=environment_count, env_spacing=(1.0, 1.0))

        self._schema = self._compute_schema()
        self._maximum_episode_steps = maximum_episode_steps
        self._episode_step_count = torch.zeros(environment_count, device=device, dtype=torch.int64)

    @property
    def schema(self) -> EnvironmentSchema:
        return self._schema

    def reset(self) -> TensorDictBase:
        self._engine.reset()

        self._episode_step_count.zero_()

        generalized_positions, generalized_velocities = self._get_character_state()
        return self._get_observation(generalized_positions, generalized_velocities)

    def reset_done(self, done: torch.Tensor) -> TensorDictBase:
        # TODO need to profile to see if this is a bottleneck
        # Could optimize or do some kind of manual reset when doing some deep-mimic style learning
        environment_indices = done.reshape(-1)
        self._engine.reset(environment_indices=environment_indices)
        self._episode_step_count.masked_fill_(done, 0)

        generalized_positions, generalized_velocities = self._get_character_state()
        return self._get_observation(generalized_positions, generalized_velocities)

    def step(self, action: TensorDictBase) -> tuple[TensorDictBase, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        self._character.apply_action(action)
        # This accessor evaluates the controller against the current state, so
        # sample it before advancing the state that the action applies to.
        control_forces = self._character.get_control_forces()
        for _ in range(self._engine.simulation_steps_per_control_step):
            self._engine.step()

        generalized_positions, generalized_velocities = self._get_character_state()
        observation = self._get_observation(generalized_positions, generalized_velocities)
        task_step = self._task.step(generalized_positions, generalized_velocities, control_forces, action)

        self._episode_step_count.add_(1)
        truncated = self._episode_step_count >= self._maximum_episode_steps

        return observation, task_step.reward, task_step.terminal, truncated, task_step.transition_metrics

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

    def _get_character_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._character.get_generalized_positions(), self._character.get_generalized_velocities()

    def _get_observation(
        self,
        generalized_positions: torch.Tensor,
        generalized_velocities: torch.Tensor,
    ) -> TensorDictBase:
        proprioception = torch.cat((generalized_positions, generalized_velocities), dim=-1)

        return TensorDict({"proprioception": proprioception}, batch_size=proprioception.shape[:-1], device=proprioception.device)
