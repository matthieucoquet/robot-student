from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from genesis.utils.geom import transform_by_quat, transform_quat_by_quat
from tensordict import TensorDict, TensorDictBase

from robot_student.engine.control_mode import ControlMode
from robot_student.environment.character_task import CharacterTask
from robot_student.environment.environment import Environment
from robot_student.environment.schema import EnvironmentSchema, TensorSchema
from robot_student.util.geometry import inverse_heading_rotation, quat_to_rot6d

if TYPE_CHECKING:
    from robot_student.engine.genesis_engine import GenesisEngine

type RootState = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class CharacterEnvironment(Environment):
    def __init__(
        self,
        engine: "GenesisEngine",
        xml_path: Path,
        environment_count: int,
        control_mode: ControlMode,
        task: CharacterTask,
        control_frequency: int,
        initial_pose: Sequence[float] | None = None,
        maximum_episode_steps: int = 1_000,
    ) -> None:
        self._count = environment_count
        self._engine = engine
        self._task = task
        self._simulation_steps_per_control_step = engine.simulation_frequency // control_frequency
        self._engine.add_ground_plane()
        self._robot = engine.add_robot(xml_path, control_mode=control_mode)

        device = engine.device
        self._engine.build_scene(environment_count=environment_count, env_spacing=(2.0, 2.0))

        if initial_pose:
            initial_pose_tensor = torch.tensor(
                initial_pose,
                dtype=torch.float32,
                device=device,
            )
            expected_shape = (self._robot.n_qs,)
            if initial_pose_tensor.shape != expected_shape:
                raise ValueError(f"initial_pose must have shape {expected_shape}, got {tuple(initial_pose_tensor.shape)}")
            batched_initial_pose = initial_pose_tensor.expand(environment_count, -1).contiguous()
            self._robot.set_default_pose(batched_initial_pose)
            self._engine.register_initial_pose()

        self._schema = self._compute_schema()
        self._maximum_episode_steps = maximum_episode_steps
        self._episode_step_count = torch.zeros(environment_count, device=device, dtype=torch.int64)
        self._global_observation = True

    @property
    def device(self) -> torch.device:
        return self._engine.device

    @property
    def count(self) -> int:
        return self._count

    @property
    def schema(self) -> EnvironmentSchema:
        return self._schema

    def reset(self) -> TensorDictBase:
        self._engine.reset()

        self._episode_step_count.zero_()

        root_state = self._robot.get_root_state()
        return self._get_character_observation(root_state)

    def reset_done(self, done: torch.Tensor) -> TensorDictBase:
        # TODO need to profile to see if this is a bottleneck
        # Could optimize or do some kind of manual reset when doing some deep-mimic style learning
        environment_indices = done.reshape(-1)
        self._engine.reset(environment_indices=environment_indices)
        self._episode_step_count.masked_fill_(done, 0)

        root_state = self._robot.get_root_state()
        return self._get_character_observation(root_state)

    def step(self, action: TensorDictBase) -> tuple[TensorDictBase, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        self._robot.apply_action(action["control"].detach())
        # This accessor evaluates the controller against the current state, so
        # sample it before advancing the state that the action applies to.
        normalized_control_forces = self._robot.get_normalized_control_forces()
        for _ in range(self._simulation_steps_per_control_step):
            self._engine.step()

        root_state = self._robot.get_root_state()
        root_position, root_rotation, root_velocity, _ = root_state
        observation = self._get_character_observation(root_state)
        joint_positions = self._robot.get_joint_dof_positions()
        task_step = self._task.step(root_position, root_rotation, root_velocity, joint_positions, normalized_control_forces)

        self._episode_step_count.add_(1)
        truncated = self._episode_step_count >= self._maximum_episode_steps

        return observation, task_step.reward, task_step.terminal, truncated, task_step.transition_metrics

    def _compute_schema(self) -> EnvironmentSchema:
        observation_type = torch.float32
        root_observation_size = 1 + 6 + 3 + 3
        proprioception_size = root_observation_size + 2 * self._robot.n_joint_dofs

        return EnvironmentSchema(
            observations={
                "proprioception": TensorSchema(
                    shape=(proprioception_size,),
                    data_type=observation_type,
                )
            },
            actions={"control": self._robot.get_action_schema()},
        )

    def _get_observation(self) -> TensorDictBase:
        return self._get_character_observation()

    def _get_character_observation(self, root_state: RootState) -> TensorDictBase:
        root_position, root_rotation, root_velocity, root_angular_velocity = root_state
        joint_positions = self._robot.get_joint_dof_positions()
        generalized_velocities = self._robot.get_generalized_velocities()
        joint_velocities = generalized_velocities[..., self._robot.n_root_dofs :]

        root_height = root_position[..., 2:3]
        if self._global_observation:
            root_rotation = quat_to_rot6d(root_rotation)
        else:
            inverse_heading = inverse_heading_rotation(root_rotation)
            local_root_rotation = transform_quat_by_quat(root_rotation, inverse_heading)
            root_rotation = quat_to_rot6d(local_root_rotation)
            root_velocity = transform_by_quat(root_velocity, inverse_heading)
            root_angular_velocity = transform_by_quat(root_angular_velocity, inverse_heading)

        proprioception_components = [
            root_height,
            root_rotation,
            root_velocity,
            root_angular_velocity,
            joint_positions,  # TODO: mimickit use 6D for each joint, relative to the rest/initial pose
            joint_velocities,
        ]
        proprioception = torch.cat(proprioception_components, dim=-1)
        return TensorDict({"proprioception": proprioception}, batch_size=proprioception.shape[:-1], device=proprioception.device)
