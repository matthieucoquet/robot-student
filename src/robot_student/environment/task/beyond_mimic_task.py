from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch
from genesis.utils.geom import inv_quat, inv_transform_by_quat, transform_by_quat, transform_quat_by_quat

from robot_student.engine.robot import Robot
from robot_student.engine.robot_state import RobotState
from robot_student.environment.schema import TensorSchema
from robot_student.environment.sensors import RotationSensor, Sensor, SensorConfiguration
from robot_student.environment.task.motion_tracking_task import MotionTrackingTask, ResetPerturbationConfiguration
from robot_student.environment.task.task import TaskEvaluation
from robot_student.motion import MotionLibrary
from robot_student.util.geometry import inverse_heading_rotation, quat_to_rot6d, quat_to_rotation_vector


@dataclass(frozen=True, kw_only=True, slots=True)
class BeyondMimicSensors:
    root_velocity: Sensor
    root_angular_velocity: Sensor
    joint_dof_positions: Sensor
    joint_dof_velocities: Sensor
    anchor_position: Sensor
    anchor_rotation: RotationSensor


@dataclass(frozen=True, kw_only=True, slots=True)
class BeyondMimicSensorConfiguration:
    root_velocity: SensorConfiguration = field(default_factory=SensorConfiguration)
    root_angular_velocity: SensorConfiguration = field(default_factory=SensorConfiguration)
    joint_dof_positions: SensorConfiguration = field(default_factory=SensorConfiguration)
    joint_dof_velocities: SensorConfiguration = field(default_factory=SensorConfiguration)
    anchor_position: SensorConfiguration = field(default_factory=SensorConfiguration)
    anchor_rotation: SensorConfiguration = field(default_factory=SensorConfiguration)

    def create_sensors(self) -> BeyondMimicSensors:
        """Create a fresh set of sensors for each task using this configuration."""
        return BeyondMimicSensors(
            root_velocity=self.root_velocity.create_sensor(Sensor),
            root_angular_velocity=self.root_angular_velocity.create_sensor(Sensor),
            joint_dof_positions=self.joint_dof_positions.create_sensor(Sensor),
            joint_dof_velocities=self.joint_dof_velocities.create_sensor(Sensor),
            anchor_position=self.anchor_position.create_sensor(Sensor),
            anchor_rotation=self.anchor_rotation.create_sensor(RotationSensor),
        )


# Differences from upstream BeyondMimic: motion completion ends the episode;
# observation noise perturbs measurements before feature construction, rather than observation terms directly.
class BeyondMimicTask(MotionTrackingTask):
    def __init__(
        self,
        xml_path: Path,
        motion_library: MotionLibrary,
        show_reference_motion: bool = False,
        reference_motion_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        anchor_link_name: str = "pelvis",
        soft_joint_position_limit_factor: float = 0.9,
        key_link_names: Sequence[str] = (),
        end_effector_link_names: tuple[str, ...] = (
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
        ),
        contact_force_threshold: float = 1.0,
        reset_perturbation_configuration: ResetPerturbationConfiguration | None = None,
        sensor_configuration: BeyondMimicSensorConfiguration | None = None,
    ) -> None:
        super().__init__(
            xml_path=xml_path,
            motion_library=motion_library,
            key_link_names=key_link_names,
            show_reference_motion=show_reference_motion,
            reference_motion_offset=reference_motion_offset,
            reset_perturbation_configuration=reset_perturbation_configuration,
            soft_joint_position_limit_factor=soft_joint_position_limit_factor,
        )
        self._anchor_link_name = anchor_link_name
        self._end_effector_link_names = end_effector_link_names
        self._contact_force_threshold = contact_force_threshold
        self._sensors = (sensor_configuration or BeyondMimicSensorConfiguration()).create_sensors()

    def initialize(
        self,
        *,
        robot: Robot,
        simulation_steps_per_control_step: int,
    ) -> None:
        super().initialize(
            robot=robot,
            simulation_steps_per_control_step=simulation_steps_per_control_step,
        )
        self._anchor_link_index = self._robot.get_link_indices([self._anchor_link_name])[0]
        end_effector_link_indices = self._robot.get_link_indices(self._end_effector_link_names)
        self._end_effector_link_indices = torch.tensor(end_effector_link_indices, dtype=torch.int64, device=self._key_link_indices.device)
        self._penalized_contact_link_indices = torch.tensor(
            [index for index in range(self._robot.n_links) if index not in end_effector_link_indices],
            dtype=torch.int64,
            device=self._key_link_indices.device,
        )
        for sensor, value in self._sensor_values():
            sensor.initialize(value)

    def _sensor_values(self) -> tuple[tuple[Sensor, torch.Tensor], ...]:
        state = self._state
        return (
            (self._sensors.root_velocity, state.root_velocity),
            (self._sensors.root_angular_velocity, state.root_angular_velocity),
            (self._sensors.joint_dof_positions, state.joint_dof_positions),
            (self._sensors.joint_dof_velocities, state.joint_dof_velocities),
            (self._sensors.anchor_position, state.world_link_positions[..., self._anchor_link_index, :]),
            (self._sensors.anchor_rotation, state.world_link_rotations[..., self._anchor_link_index, :]),
        )

    @torch.no_grad()
    def reset(self, environment_indices: torch.Tensor) -> None:
        super().reset(environment_indices)
        for sensor, value in self._sensor_values():
            sensor.reset(value, environment_indices)

    @torch.no_grad()
    def measure_state(self) -> None:
        super().measure_state()
        for sensor, value in self._sensor_values():
            sensor.update(value)

    def get_schema(self) -> dict[str, TensorSchema]:
        joint_count = self._robot.n_joint_dofs
        link_count = self._key_link_indices.numel()

        motion_phase_size = 2 * joint_count
        anchor_error_size = 9
        root_velocity_size = 6
        joint_size = 2 * joint_count
        link_size = 9 * link_count
        previous_action_size = self._robot.n_controlled_dofs
        actor_size = motion_phase_size + anchor_error_size + root_velocity_size + joint_size + previous_action_size
        critic_size = actor_size + link_size

        sizes = {
            "actor": actor_size,
            "critic": critic_size,
        }
        return {key: TensorSchema(shape=(size,), data_type=torch.float32) for key, size in sizes.items()}

    def observation(self, *, previous_action: torch.Tensor) -> dict[str, torch.Tensor]:
        state = self._state
        motion_command = torch.cat(
            (self._reference_state.joint_dof_positions, self._reference_state.joint_dof_velocities),
            dim=-1,
        )

        actor_components = self._observation_components(
            anchor_position=self._sensors.anchor_position.observation,
            inverse_anchor_rotation=inv_quat(self._sensors.anchor_rotation.observation),
            root_velocity=self._sensors.root_velocity.observation,
            root_angular_velocity=self._sensors.root_angular_velocity.observation,
            joint_dof_positions=self._sensors.joint_dof_positions.observation,
            joint_dof_velocities=self._sensors.joint_dof_velocities.observation,
        )

        anchor_position = state.world_link_positions[..., self._anchor_link_index, :]
        inverse_anchor_rotation = inv_quat(state.world_link_rotations[..., self._anchor_link_index, :])
        critic_components = self._observation_components(
            anchor_position=anchor_position,
            inverse_anchor_rotation=inverse_anchor_rotation,
            root_velocity=state.root_velocity,
            root_angular_velocity=state.root_angular_velocity,
            joint_dof_positions=state.joint_dof_positions,
            joint_dof_velocities=state.joint_dof_velocities,
        )

        world_link_positions = state.world_link_positions.index_select(-2, self._key_link_indices)
        world_link_rotations = state.world_link_rotations.index_select(-2, self._key_link_indices)
        link_positions = transform_by_quat(
            world_link_positions - anchor_position.unsqueeze(-2), inverse_anchor_rotation.unsqueeze(-2)
        ).flatten(start_dim=-2)
        link_rotations = quat_to_rot6d(transform_quat_by_quat(world_link_rotations, inverse_anchor_rotation.unsqueeze(-2))).flatten(
            start_dim=-2
        )

        return {
            "actor": torch.cat((motion_command, *actor_components, previous_action), dim=-1),
            "critic": torch.cat((motion_command, *critic_components, link_positions, link_rotations, previous_action), dim=-1),
        }

    def _observation_components(
        self,
        *,
        anchor_position: torch.Tensor,
        inverse_anchor_rotation: torch.Tensor,
        root_velocity: torch.Tensor,
        root_angular_velocity: torch.Tensor,
        joint_dof_positions: torch.Tensor,
        joint_dof_velocities: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Build shared actor and critic features from sensor readings or clean state values."""
        reference_anchor_position = self._reference_state.world_link_positions[..., self._anchor_link_index, :]
        reference_anchor_rotation = self._reference_state.world_link_rotations[..., self._anchor_link_index, :]
        anchor_position_error = transform_by_quat(reference_anchor_position - anchor_position, inverse_anchor_rotation)
        anchor_rotation_error = quat_to_rot6d(transform_quat_by_quat(reference_anchor_rotation, inverse_anchor_rotation))
        root_linear_velocities = inv_transform_by_quat(root_velocity, self._state.root_rotation)
        root_angular_velocities = inv_transform_by_quat(root_angular_velocity, self._state.root_rotation)
        joint_positions = joint_dof_positions - self._robot.default_joint_positions

        return (
            anchor_position_error,
            anchor_rotation_error,
            root_linear_velocities,
            root_angular_velocities,
            joint_positions,
            joint_dof_velocities,
        )

    def _compute_reward(
        self,
        state: RobotState,
        reference: RobotState,
        *,
        current_action: torch.Tensor,
        previous_action: torch.Tensor,
        contact_forces: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        anchor_position = state.world_link_positions[..., self._anchor_link_index, :]
        reference_anchor_position = reference.world_link_positions[..., self._anchor_link_index, :]
        anchor_rotation = state.world_link_rotations[..., self._anchor_link_index, :]
        reference_anchor_rotation = reference.world_link_rotations[..., self._anchor_link_index, :]

        anchor_position_error = (anchor_position - reference_anchor_position).square().sum(dim=-1)
        anchor_position_reward = torch.exp(-anchor_position_error / 0.3**2)

        anchor_rotation_difference = transform_quat_by_quat(inv_quat(reference_anchor_rotation), anchor_rotation)
        anchor_rotation_error = quat_to_rotation_vector(anchor_rotation_difference).square().sum(dim=-1)
        anchor_rotation_reward = torch.exp(-anchor_rotation_error / 0.4**2)

        # ignore heading rotation for body alignment
        alignment_rotation = inv_quat(inverse_heading_rotation(anchor_rotation_difference)).unsqueeze(-2)
        # try to match the reference height
        alignment_position = torch.cat((anchor_position[..., :2], reference_anchor_position[..., 2:3]), dim=-1).unsqueeze(-2)

        world_link_positions = state.world_link_positions.index_select(-2, self._key_link_indices)
        world_reference_link_positions = reference.world_link_positions.index_select(-2, self._key_link_indices)
        aligned_reference_link_positions = alignment_position + transform_by_quat(
            world_reference_link_positions - reference_anchor_position.unsqueeze(-2), alignment_rotation
        )
        body_position_error = (world_link_positions - aligned_reference_link_positions).square().sum(dim=-1).mean(dim=-1)
        body_position_reward = torch.exp(-body_position_error / 0.3**2)

        world_link_rotations = state.world_link_rotations.index_select(-2, self._key_link_indices)
        world_reference_link_rotations = reference.world_link_rotations.index_select(-2, self._key_link_indices)
        aligned_reference_link_rotations = transform_quat_by_quat(world_reference_link_rotations, alignment_rotation)
        body_rotation_difference = transform_quat_by_quat(inv_quat(aligned_reference_link_rotations), world_link_rotations)
        body_rotation_error = quat_to_rotation_vector(body_rotation_difference).square().sum(dim=-1).mean(dim=-1)
        body_rotation_reward = torch.exp(-body_rotation_error / 0.4**2)

        world_link_linear_velocities = state.world_link_linear_velocities.index_select(-2, self._key_link_indices)
        world_reference_link_linear_velocities = reference.world_link_linear_velocities.index_select(-2, self._key_link_indices)
        body_linear_velocity_error = (
            (world_link_linear_velocities - world_reference_link_linear_velocities).square().sum(dim=-1).mean(dim=-1)
        )
        body_linear_velocity_reward = torch.exp(-body_linear_velocity_error / 1.0**2)

        world_link_angular_velocities = state.world_link_angular_velocities.index_select(-2, self._key_link_indices)
        world_reference_link_angular_velocities = reference.world_link_angular_velocities.index_select(-2, self._key_link_indices)
        body_angular_velocity_error = (
            (world_link_angular_velocities - world_reference_link_angular_velocities).square().sum(dim=-1).mean(dim=-1)
        )
        body_angular_velocity_reward = torch.exp(-body_angular_velocity_error / 3.14**2)

        action_change = (current_action - previous_action) / self._robot.control_action_scale
        action_change_penalty = action_change.square().sum(dim=-1)

        joint_position_limit_penalty = (
            (self._soft_joint_position_lower_bounds - state.joint_dof_positions).clamp_min(0.0)
            + (state.joint_dof_positions - self._soft_joint_position_upper_bounds).clamp_min(0.0)
        ).sum(dim=-1)

        penalized_contact_forces = contact_forces.index_select(-2, self._penalized_contact_link_indices)
        contact_force_magnitudes = torch.linalg.vector_norm(penalized_contact_forces, dim=-1)
        undesired_contact_penalty = (contact_force_magnitudes > self._contact_force_threshold).sum(
            dim=-1, dtype=state.joint_dof_positions.dtype
        )

        reward = (
            0.5 * anchor_position_reward
            + 0.5 * anchor_rotation_reward
            + body_position_reward
            + body_rotation_reward
            + body_linear_velocity_reward
            + body_angular_velocity_reward
            - 0.1 * action_change_penalty
            - 10.0 * joint_position_limit_penalty
            - 0.1 * undesired_contact_penalty
        )
        reward_components = (
            anchor_position_reward,
            anchor_rotation_reward,
            body_position_reward,
            body_rotation_reward,
            body_linear_velocity_reward,
            body_angular_velocity_reward,
            action_change_penalty,
            joint_position_limit_penalty,
            undesired_contact_penalty,
        )
        return reward, reward_components

    def _compute_terminal(self, state: RobotState, reference: RobotState) -> torch.Tensor:
        anchor_height = state.world_link_positions[..., self._anchor_link_index, 2]
        reference_anchor_height = reference.world_link_positions[..., self._anchor_link_index, 2]
        anchor_height_error = (anchor_height - reference_anchor_height).abs()

        end_effector_heights = state.world_link_positions[..., 2].index_select(-1, self._end_effector_link_indices)
        reference_end_effector_heights = reference.world_link_positions[..., 2].index_select(-1, self._end_effector_link_indices)
        end_effector_height_errors = (end_effector_heights - reference_end_effector_heights).abs()

        anchor_rotation = state.world_link_rotations[..., self._anchor_link_index, :]
        reference_anchor_rotation = reference.world_link_rotations[..., self._anchor_link_index, :]

        anchor_gravity_z = 2.0 * anchor_rotation[..., 1:3].square().sum(dim=-1) - 1.0
        reference_anchor_gravity_z = 2.0 * reference_anchor_rotation[..., 1:3].square().sum(dim=-1) - 1.0
        anchor_tilt_error = (anchor_gravity_z - reference_anchor_gravity_z).abs()

        return (anchor_height_error > 0.25) | (end_effector_height_errors > 0.25).any(dim=-1) | (anchor_tilt_error > 0.8)

    def evaluate_step(
        self,
        *,
        current_action: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> TaskEvaluation:
        state = self._state
        reward, reward_components = self._compute_reward(
            state,
            self._reference_state,
            current_action=current_action,
            previous_action=previous_action,
            contact_forces=self._robot.get_links_net_contact_force(),
        )
        (
            anchor_position_reward,
            anchor_rotation_reward,
            body_position_reward,
            body_rotation_reward,
            body_linear_velocity_reward,
            body_angular_velocity_reward,
            action_change_penalty,
            joint_position_limit_penalty,
            undesired_contact_penalty,
        ) = reward_components

        terminal = self._compute_terminal(state, self._reference_state)
        self._reference_robot.record_failures(terminal)
        terminal = terminal | self._motion_finished
        return TaskEvaluation(
            reward=reward,
            terminal=terminal,
            transition_metrics={
                "task/anchor_position_reward_mean": anchor_position_reward.mean(),
                "task/anchor_rotation_reward_mean": anchor_rotation_reward.mean(),
                "task/body_position_reward_mean": body_position_reward.mean(),
                "task/body_rotation_reward_mean": body_rotation_reward.mean(),
                "task/body_linear_velocity_reward_mean": body_linear_velocity_reward.mean(),
                "task/body_angular_velocity_reward_mean": body_angular_velocity_reward.mean(),
                "task/action_change_penalty_mean": action_change_penalty.mean(),
                "task/joint_position_limit_penalty_mean": joint_position_limit_penalty.mean(),
                "task/undesired_contact_penalty_mean": undesired_contact_penalty.mean(),
            },
        )
