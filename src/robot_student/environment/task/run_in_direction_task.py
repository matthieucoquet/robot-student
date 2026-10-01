import math
from collections.abc import Sequence

import torch
from genesis.utils.geom import inv_quat, inv_transform_by_quat, transform_by_quat, transform_quat_by_quat

from robot_student.engine.robot import Robot
from robot_student.engine.robot_state import RobotState
from robot_student.environment.schema import TensorSchema
from robot_student.environment.task.task import Task, TaskFeedback
from robot_student.util.geometry import heading_angle, inverse_heading_rotation, quat_to_rot6d


class RunInDirectionTask(Task):
    def __init__(
        self,
        device: torch.device,
        imu_link_name: str,
        default_joint_positions: Sequence[float],
        direction: tuple[float, float] = (1.0, 0.0),
        target_speed: float = 1.0,
        target_speed_weight: float = 1.0,
        target_height: float = 0.75,
        target_height_weight: float = 1.0,
        facing_direction_weight: float = 0.5,
        control_cost_weight: float = 0.5,
        pose_cost_weight: float = 0.1,
        height_range: tuple[float, float] = (0.2, 1.0),
    ) -> None:
        direction_norm = math.hypot(*direction)
        if not math.isclose(direction_norm, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"direction must be normalized, got norm {direction_norm}")

        self._imu_link_name = imu_link_name
        self._world_gravity = torch.tensor((0.0, 0.0, -1.0), device=device, dtype=torch.float32)
        self._direction = torch.tensor(direction, device=device, dtype=torch.float32)
        self._direction_heading = math.atan2(direction[1], direction[0])
        self._target_velocity = self._direction * target_speed
        self._target_speed_weight = target_speed_weight
        self._target_height = target_height
        self._target_height_weight = target_height_weight
        self._facing_direction_weight = facing_direction_weight
        self._control_cost_weight = control_cost_weight
        self._pose_cost_weight = pose_cost_weight
        self._default_joint_positions = torch.tensor(default_joint_positions, device=device, dtype=torch.float32)
        self._minimum_healthy_height, self._maximum_healthy_height = height_range

    def get_schema(self) -> dict[str, TensorSchema]:
        joint_count = self._robot.n_joint_dofs
        link_count = self._key_link_indices.numel()

        imu_states = 6
        joint_size = 2 * joint_count
        link_size = 9 * link_count
        previous_action_size = self._robot.n_controlled_dofs
        actor_size = imu_states + joint_size + previous_action_size
        critic_size = actor_size + link_size

        sizes = {
            "actor": actor_size,
            "critic": critic_size,
        }
        return {key: TensorSchema(shape=(size,), data_type=torch.float32) for key, size in sizes.items()}

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
        self._global_observation = global_observation

        self._imu_link_index = self._robot.get_link_indices([self._imu_link_name])[0]

    def reset(self, environment_indices: torch.Tensor) -> None:
        pass

    def observation(self, robot_state: RobotState, *, noisy_state: RobotState, previous_action: torch.Tensor) -> dict[str, torch.Tensor]:

        for noisy_observation in [True, False]:
            observed_state = noisy_state if noisy_observation else robot_state

            imu_rotation = observed_state.world_link_rotations[..., self._imu_link_index, :]
            inverse_imu_rotation = inv_quat(imu_rotation)
            imu_angular_velocity = observed_state.world_link_angular_velocities[..., self._imu_link_index, :]
            imu_angular_velocity = transform_by_quat(imu_angular_velocity, inverse_imu_rotation)
            projected_gravity = transform_by_quat(self._world_gravity, inverse_imu_rotation)

            joint_positions = observed_state.joint_dof_positions - self._robot.default_joint_positions
            joint_velocities = observed_state.joint_dof_velocities

            if noisy_observation:
                actor = torch.cat(
                    (imu_angular_velocity, projected_gravity, joint_positions, joint_velocities, previous_action),
                    dim=-1,
                )
            else:
                # For now the additional variable are different than microduck rl
                world_link_positions = robot_state.world_link_positions.index_select(-2, self._key_link_indices)
                world_link_rotations = robot_state.world_link_rotations.index_select(-2, self._key_link_indices)
                imu_link_position = observed_state.world_link_positions[..., self._imu_link_index, :]
                link_positions = transform_by_quat(
                    world_link_positions - imu_link_position.unsqueeze(-2), inverse_imu_rotation.unsqueeze(-2)
                ).flatten(start_dim=-2)
                link_rotations = quat_to_rot6d(transform_quat_by_quat(world_link_rotations, inverse_imu_rotation.unsqueeze(-2))).flatten(
                    start_dim=-2
                )
                critic = torch.cat(
                    (
                        imu_angular_velocity,
                        projected_gravity,
                        joint_positions,
                        joint_velocities,
                        previous_action,
                        link_positions,
                        link_rotations,
                    ),
                    dim=-1,
                )
        return {
            "actor": actor,
            "critic": critic,
        }

    def compute_feedback(
        self,
        state: RobotState,
        *,
        normalized_control_forces: torch.Tensor,
        current_action: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> TaskFeedback:
        root_height = state.root_position[..., 2]
        root_height_is_healthy = root_height >= self._minimum_healthy_height
        root_height_is_healthy.logical_and_(root_height <= self._maximum_healthy_height)
        terminal = ~root_height_is_healthy

        planar_velocity = state.root_velocity[..., :2]
        target_velocity_error = torch.linalg.vector_norm(planar_velocity - self._target_velocity, dim=-1)
        target_velocity_reward = target_velocity_error.mul(10.0).neg_().exp_()

        target_height_error = root_height - self._target_height
        target_height_reward = target_height_error.clamp(max=0.0).abs_().mul_(25.0).neg_().exp_()

        facing_direction_reward = torch.cos(heading_angle(state.root_rotation) - self._direction_heading)

        control_cost = torch.mean(normalized_control_forces.square(), dim=-1)
        pose_cost = torch.mean((state.joint_dof_positions - self._default_joint_positions).square(), dim=-1)
        stay_alive_reward = root_height_is_healthy * 0.05
        reward = (
            stay_alive_reward
            + self._target_speed_weight * target_velocity_reward
            + self._target_height_weight * target_height_reward
            + self._facing_direction_weight * facing_direction_reward
            - self._control_cost_weight * control_cost
            - self._pose_cost_weight * pose_cost
        )

        return TaskFeedback(
            reward=reward,
            terminal=terminal,
            transition_metrics={
                "task/target_velocity_error_mean": target_velocity_error.mean(),
                "task/target_height_error_mean": target_height_error.abs().mean(),
                "task/facing_direction_reward_mean": facing_direction_reward.mean(),
                "task/root_height_mean": root_height.mean(),
                "task/control_cost_mean": control_cost.mean(),
                "task/pose_cost_mean": pose_cost.mean(),
            },
        )
