from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from genesis.utils.geom import inv_quat, transform_by_quat, transform_quat_by_quat
from tensordict import TensorDict, TensorDictBase

from robot_student.engine.control_mode import ControlMode
from robot_student.engine.kinematic_robot import RobotState
from robot_student.environment.character_environment import CharacterEnvironment
from robot_student.environment.character_task import CharacterTask, CharacterTaskStep
from robot_student.environment.schema import EnvironmentSchema, TensorSchema
from robot_student.motion import MotionLibrary, ReferenceRobot
from robot_student.util.geometry import inverse_heading_rotation, quat_to_rot6d, quat_to_rotation_vector

if TYPE_CHECKING:
    from robot_student.engine.genesis_engine import GenesisEngine


class DeepMimicTask(CharacterTask):
    def __init__(self, device: torch.device, joint_reward_weight: Sequence[float]) -> None:
        super().__init__()
        self._joint_reward_weight = torch.tensor(joint_reward_weight, dtype=torch.float32, device=device)

    def set_key_link_indices(self, key_link_indices):
        self._key_link_indices = key_link_indices

    def compute_reward(
        self,
        state: RobotState,
        reference: RobotState,
    ) -> torch.Tensor:
        position_difference = state.joint_dof_positions - reference.joint_dof_positions
        position_error = torch.sum(self._joint_reward_weight * position_difference.square(), dim=-1)
        pose_reward = torch.exp(-0.25 * position_error)

        velocity_difference = state.joint_dof_velocities - reference.joint_dof_velocities
        velocity_error = torch.sum(self._joint_reward_weight * velocity_difference.square(), dim=-1)
        velocity_reward = torch.exp(-0.01 * velocity_error)

        key_link_positions = state.world_link_positions.index_select(-2, self._key_link_indices)
        reference_key_link_positions = reference.world_link_positions.index_select(-2, self._key_link_indices)
        key_link_positions -= state.root_position.unsqueeze(-2)
        reference_key_link_positions -= reference.root_position.unsqueeze(-2)

        root_position_difference = state.root_position - reference.root_position
        root_position_error = torch.sum(root_position_difference.square(), dim=-1)

        root_rotation_difference = transform_quat_by_quat(inv_quat(state.root_rotation), reference.root_rotation)
        root_rotation_error = torch.sum(quat_to_rotation_vector(root_rotation_difference).square(), dim=-1)
        root_pose_reward = torch.exp(-5.0 * (root_position_error + 0.1 * root_rotation_error))

        root_velocity_error = torch.sum((state.root_velocity - reference.root_velocity).square(), dim=-1)
        root_angular_velocity_error = torch.sum((state.root_angular_velocity - reference.root_angular_velocity).square(), dim=-1)
        root_velocity_reward = torch.exp(-(root_velocity_error + 0.1 * root_angular_velocity_error))

        key_position_error = torch.sum((key_link_positions - reference_key_link_positions).square(), dim=(-2, -1))
        key_position_reward = torch.exp(-10.0 * key_position_error)

        return 0.5 * pose_reward + 0.1 * velocity_reward + 0.15 * root_pose_reward + 0.1 * root_velocity_reward + 0.15 * key_position_reward

    def compute_terminal(self, state: RobotState, reference: RobotState):
        link_differences = state.world_link_positions - reference.world_link_positions
        link_distances = torch.sum(link_differences.square(), dim=-1)
        link_distances = torch.max(link_distances, dim=-1)[0]

        return link_distances > 1.0

    def step(
        self,
        state: RobotState,
        reference: RobotState,
        **_,
    ) -> CharacterTaskStep:
        reward = self.compute_reward(state, reference)
        terminal = self.compute_terminal(state, reference)
        return CharacterTaskStep(reward=reward, terminal=terminal, transition_metrics={})


class MotionTrackingEnvironment(CharacterEnvironment):
    def __init__(
        self,
        engine: "GenesisEngine",
        motion_library: MotionLibrary,
        xml_path: Path,
        environment_count: int,
        control_mode: ControlMode,
        task: CharacterTask,
        control_frequency: int,
        initial_pose: Sequence[float],
        key_link_names: Sequence[str],
        target_steps: list,
        maximum_episode_steps: int = 1_000,
    ) -> None:

        control_timestep = 1 / control_frequency
        self._reference_robot = ReferenceRobot(environment_count, motion_library, control_timestep, engine.device)
        self._target_steps = torch.tensor(target_steps, dtype=torch.float32, device=engine.device)

        super().__init__(
            engine,
            xml_path,
            environment_count,
            control_mode,
            task,
            control_frequency,
            initial_pose=initial_pose,
            key_link_names=key_link_names,
            maximum_episode_steps=maximum_episode_steps,
        )
        self._task.set_key_link_indices(self._key_link_indices)

    def _compute_schema(self) -> EnvironmentSchema:
        observation_type = torch.float32
        root_observation_size = 1 + 6 + 3 + 3
        key_link_position_size = 3 * self._key_link_indices.numel()
        proprioception_size = root_observation_size + 2 * self._robot.n_joint_dofs + key_link_position_size
        target_step_size = 3 + 6 + self._robot.n_joint_dofs + key_link_position_size
        target_size = self._target_steps.numel() * target_step_size

        return EnvironmentSchema(
            observations={
                "proprioception": TensorSchema(
                    shape=(proprioception_size,),
                    data_type=observation_type,
                ),
                "target": TensorSchema(
                    shape=(target_size,),
                    data_type=observation_type,
                ),
            },
            actions={"control": self._get_control_schema()},
        )

    def reset(self) -> TensorDictBase:
        self._episode_step_count.zero_()
        reference_state = self._reference_robot.reset()
        self._state = self._robot.set_state(reference_state)
        return self._get_observation()

    def reset_done(self, done: torch.Tensor) -> TensorDictBase:
        self._episode_step_count.masked_fill_(done, 0)
        environment_indices = done.reshape(-1).nonzero().reshape(-1)

        if environment_indices.numel() > 0:
            # self._engine.reset(environment_indices=environment_indices)  # Probably not needed to call genesis reset?
            reference_state = self._reference_robot.reset(environment_indices)
            reset_state = self._robot.set_state(reference_state, environment_indices=environment_indices)
            self._state.copy_environments_(environment_indices, reset_state)

        return self._get_observation()

    def step(self, action: TensorDictBase) -> tuple[TensorDictBase, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        self._robot.apply_control(action["control"].detach())
        for _ in range(self._simulation_steps_per_control_step):
            self._engine.step()

        self._state = self._robot.get_state()
        self._episode_step_count.add_(1)

        observation = self._get_observation()
        reference_state, motion_finished = self._reference_robot.get_state(self._episode_step_count)
        task_step = self._task.step(self._state, reference=reference_state)

        terminal = torch.logical_or(task_step.terminal, motion_finished)
        truncated = self._episode_step_count >= self._maximum_episode_steps

        return observation, task_step.reward, terminal, truncated, task_step.transition_metrics

    def _get_observation(self) -> TensorDictBase:
        observation = self._get_character_observation()
        target = self._get_target_observation()
        observation.update(target)
        return observation

    def _get_target_observation(self) -> TensorDictBase:
        targets = self._reference_robot.get_target_states(self._episode_step_count, self._target_steps)

        inverse_headings = inverse_heading_rotation(targets.root_rotation)
        key_link_positions = targets.world_link_positions.index_select(-2, self._key_link_indices)
        heading_relative_key_link_positions = transform_by_quat(
            key_link_positions - targets.root_position.unsqueeze(-2),
            inverse_headings.unsqueeze(-2),
        ).flatten(start_dim=-2)

        if self._global_observation:
            target_root_position = targets.root_position - self._state.root_position.unsqueeze(-2)
            target_root_rotation = targets.root_rotation
        else:
            target_root_position = targets.root_position - targets.root_position[..., :1, :]

            reference_inverse_heading = inverse_headings[..., :1, :]
            target_root_position = transform_by_quat(target_root_position, reference_inverse_heading)
            target_root_position[..., 2] = targets.root_position[..., 2]  # We keep the height

            target_root_rotation = transform_quat_by_quat(targets.root_rotation, reference_inverse_heading)

        target_root_rotation = quat_to_rot6d(target_root_rotation)

        target_components = [
            target_root_position,
            target_root_rotation,
            targets.joint_dof_positions,  # same as character mimickit use 6D for each joint, for now we use 1D
            heading_relative_key_link_positions,
        ]
        target = torch.cat(target_components, dim=-1).flatten(start_dim=-2)
        return TensorDict({"target": target}, batch_size=target.shape[:-1], device=target.device)
