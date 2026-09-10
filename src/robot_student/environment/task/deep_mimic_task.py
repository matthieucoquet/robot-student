from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from genesis.utils.geom import inv_quat, transform_by_quat, transform_quat_by_quat

from robot_student.engine.genesis_engine import GenesisEngine
from robot_student.engine.kinematic_robot import RobotState
from robot_student.engine.robot import Robot
from robot_student.environment.schema import EnvironmentSchema, TensorSchema
from robot_student.environment.task.task import Task, TaskFeedback
from robot_student.motion import MotionLibrary, ReferenceRobot
from robot_student.util.geometry import inverse_heading_rotation, quat_to_rot6d, quat_to_rotation_vector


class DeepMimicTask(Task):
    def __init__(
        self,
        engine: GenesisEngine,
        environment_count: int,
        xml_path: Path,
        motion_library: MotionLibrary,
        target_steps: Sequence[float],
        joint_reward_weight: Sequence[float],
        random_reference_sampling: bool = False,
        show_reference_motion: bool = False,
        reference_motion_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        super().__init__()

        self._target_steps = torch.tensor(target_steps, dtype=torch.float32, device=engine.device)
        self._random_reference_sampling = random_reference_sampling

        self._kinematic_robot = None
        self._show_reference_motion = show_reference_motion
        if self._show_reference_motion:
            self._kinematic_robot = engine.add_kinematic_robot(
                xml_path,
                position_offset=(0.0, 0.0, 0.0),
                color=(0.15, 0.55, 1.0, 0.7),
                name="reference_robot",
            )

        self._reference_robot = ReferenceRobot(
            environment_count,
            motion_library,
            engine.time_step,
            engine.device,
            kinematic_robot=self._kinematic_robot,
            display_offset=reference_motion_offset,
        )

        self._joint_reward_weight = torch.tensor(joint_reward_weight, dtype=torch.float32, device=engine.device)

    def initialize(
        self,
        *,
        robot: Robot,
        key_link_indices: torch.Tensor,
        simulation_steps_per_control_step: int,
        global_observation: bool,
    ) -> None:
        self._robot = robot
        self._key_link_indices = key_link_indices
        self._simulation_steps_per_control_step = simulation_steps_per_control_step
        self._global_observation = global_observation

    def reset(self, environment_indices: torch.Tensor) -> None:
        reference_state = self._reference_robot.reset(
            random_sampling=self._random_reference_sampling,
            environment_indices=environment_indices,
        )
        self._robot.set_state(reference_state, environment_indices=environment_indices)

    def get_schema(self) -> dict[str, TensorSchema]:
        key_link_position_size = 3 * self._key_link_indices.numel()
        target_step_size = 3 + 6 + self._robot.n_joint_dofs + key_link_position_size
        target_size = self._target_steps.numel() * target_step_size

        return {
            "target": TensorSchema(
                shape=(target_size,),
                data_type=torch.float32,
            )
        }

    def observation(self, robot_state: RobotState) -> dict[str, torch.Tensor]:
        targets = self._reference_robot.get_target_states(self._simulation_steps_per_control_step, self._target_steps)

        key_link_positions = targets.world_link_positions.index_select(-2, self._key_link_indices)
        relative_key_link_positions = key_link_positions - targets.root_position.unsqueeze(-2)

        if self._global_observation:
            target_root_position = targets.root_position - robot_state.root_position.unsqueeze(-2)
            target_root_rotation = targets.root_rotation
        else:
            inverse_headings = inverse_heading_rotation(targets.root_rotation)
            relative_key_link_positions = transform_by_quat(relative_key_link_positions, inverse_headings.unsqueeze(-2))
            target_root_position = targets.root_position - targets.root_position[..., :1, :]

            reference_inverse_heading = inverse_headings[..., :1, :]
            target_root_position = transform_by_quat(target_root_position, reference_inverse_heading)
            target_root_rotation = transform_quat_by_quat(targets.root_rotation, reference_inverse_heading)

        target_root_position[..., 2] = targets.root_position[..., 2]
        target_root_rotation = quat_to_rot6d(target_root_rotation)

        target_components = [
            target_root_position,
            target_root_rotation,
            targets.joint_dof_positions,  # same as character mimickit use 6D for each joint, for now we use 1D
            relative_key_link_positions.flatten(start_dim=-2),
        ]
        target = torch.cat(target_components, dim=-1).flatten(start_dim=-2)
        return {"target": target}

    def _compute_reward(
        self,
        state: RobotState,
        reference: RobotState,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
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

        reward = (
            0.5 * pose_reward + 0.1 * velocity_reward + 0.15 * root_pose_reward + 0.1 * root_velocity_reward + 0.15 * key_position_reward
        )
        reward_components = (pose_reward, velocity_reward, root_pose_reward, root_velocity_reward, key_position_reward)
        return reward, reward_components

    def _compute_terminal(self, state: RobotState, reference: RobotState):
        link_differences = state.world_link_positions - reference.world_link_positions
        link_distances = torch.sum(link_differences.square(), dim=-1)
        link_distances = torch.max(link_distances, dim=-1)[0]

        terminal = link_distances > 1.0
        terminal.logical_or_(self._motion_finished)
        return terminal

    def step(self, is_control_step: bool) -> None:
        if self._show_reference_motion:
            self._reference_state, self._motion_finished = self._reference_robot.step(1)
        elif is_control_step:
            self._reference_state, self._motion_finished = self._reference_robot.step(self._simulation_steps_per_control_step)

    def compute_feedback(self, state: RobotState, **kwargs: Any) -> TaskFeedback:
        reward, reward_components = self._compute_reward(state, self._reference_state)
        pose_reward, velocity_reward, root_pose_reward, root_velocity_reward, key_position_reward = reward_components
        terminal = self._compute_terminal(state, self._reference_state)
        return TaskFeedback(
            reward=reward,
            terminal=terminal,
            transition_metrics={
                "task/pose_reward_mean": pose_reward.mean(),
                "task/velocity_reward_mean": velocity_reward.mean(),
                "task/root_pose_reward_mean": root_pose_reward.mean(),
                "task/root_velocity_reward_mean": root_velocity_reward.mean(),
                "task/key_position_reward_mean": key_position_reward.mean(),
            },
        )
