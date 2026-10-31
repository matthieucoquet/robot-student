from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch
from genesis.utils.geom import inv_quat, inv_transform_by_quat, transform_by_quat, transform_quat_by_quat
from tensordict import TensorClass

from robot_student.engine.robot import Robot
from robot_student.environment.schema import TensorSchema
from robot_student.environment.sensors import IMUSensor, RotationSensor, Sensor
from robot_student.environment.task.motion_tracking_task import MotionTrackingTask, ResetPerturbationConfiguration
from robot_student.environment.task.task import TaskEvaluation
from robot_student.motion import MotionLibrary
from robot_student.util.geometry import inverse_heading_rotation, quat_to_rot6d, quat_to_rotation_vector


class SimulatorObservation(TensorClass["autocast"]):
    """Clean task measurements captured at one nominal instant."""

    root_rotation: torch.Tensor
    root_velocity: torch.Tensor
    imu_rotation: torch.Tensor
    imu_angular_velocity: torch.Tensor  # World coordinates.
    servo_encoder_positions: torch.Tensor
    servo_encoder_velocities: torch.Tensor
    world_link_positions: torch.Tensor
    world_link_rotations: torch.Tensor
    world_link_linear_velocities: torch.Tensor
    world_link_angular_velocities: torch.Tensor
    contact_forces: torch.Tensor


@dataclass(frozen=True, kw_only=True, slots=True)
class BeyondMimicSensors:
    """Sensors owned by one task, with fresh instances for each omitted field."""

    root_velocity: Sensor = field(default_factory=Sensor)
    imu: IMUSensor = field(default_factory=IMUSensor)
    servo_encoder_positions: Sensor = field(default_factory=Sensor)
    servo_encoder_velocities: Sensor = field(default_factory=Sensor)
    anchor_position: Sensor = field(default_factory=Sensor)
    anchor_rotation: RotationSensor = field(default_factory=RotationSensor)


# Differences from upstream BeyondMimic: motion completion ends the episode;
# observation noise perturbs measurements before feature construction, rather than observation terms directly.
class BeyondMimicTask(MotionTrackingTask):
    """Track reference motion with encoder and IMU observations and clean measurements for the critic and costs.

    The anchor link also carries the IMU. Angular velocity and projected gravity use its frame;
    root linear velocity uses the root frame. Named measured and simulated features are kept separate;
    motion commands and previous actions are shared by the actor and critic.
    """

    actor_observation_keys = (
        "motion_command",
        "measured_anchor_position_error",
        "measured_anchor_rotation_error",
        "measured_root_linear_velocity",
        "measured_imu_angular_velocity",
        "measured_projected_gravity",
        "measured_servo_encoder_position_offsets",
        "measured_servo_encoder_velocities",
        "previous_action",
    )
    critic_observation_keys = (
        "motion_command",
        "simulated_anchor_position_error",
        "simulated_anchor_rotation_error",
        "simulated_root_linear_velocity",
        "simulated_imu_angular_velocity",
        "simulated_projected_gravity",
        "simulated_servo_encoder_position_offsets",
        "simulated_servo_encoder_velocities",
        "simulated_link_positions",
        "simulated_link_rotations",
        "previous_action",
    )

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
        sensors: BeyondMimicSensors | None = None,
        tracking_position_tolerance: float = 0.3,
        tracking_linear_velocity_tolerance: float = 1.0,
        tracking_height_threshold: float = 0.25,
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
        self._tracking_position_variance = tracking_position_tolerance**2
        self._tracking_linear_velocity_variance = tracking_linear_velocity_tolerance**2
        self._tracking_height_threshold = tracking_height_threshold
        self._sensors = sensors if sensors is not None else BeyondMimicSensors()

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
        servo_joints = self._robot.servo_joints
        self._soft_encoder_position_lower_bounds = servo_joints.select_motor_joint_values(self._soft_joint_position_lower_bounds)
        self._soft_encoder_position_upper_bounds = servo_joints.select_motor_joint_values(self._soft_joint_position_upper_bounds)
        self._anchor_link_index = self._robot.get_link_indices([self._anchor_link_name])[0]
        end_effector_link_indices = self._robot.get_link_indices(self._end_effector_link_names)
        self._end_effector_link_indices = torch.tensor(end_effector_link_indices, dtype=torch.int64, device=self._key_link_indices.device)
        self._penalized_contact_link_indices = torch.tensor(
            [index for index in range(self._robot.n_links) if index not in end_effector_link_indices],
            dtype=torch.int64,
            device=self._key_link_indices.device,
        )
        self._simulator_observation = self._read_simulator_observation()
        self._world_gravity = self._simulator_observation.imu_rotation.new_tensor((0.0, 0.0, -1.0))
        for sensor, value in self._sensor_values():
            sensor.initialize(value)
        if not self._sensors.imu.is_passthrough:
            self._sensors.imu.initialize(self._simulator_observation.imu_rotation, self._simulator_observation.imu_angular_velocity)

    def _sensor_values(self) -> tuple[tuple[Sensor, torch.Tensor], ...]:
        simulator_observation = self._simulator_observation
        return (
            (self._sensors.root_velocity, simulator_observation.root_velocity),
            (self._sensors.servo_encoder_positions, simulator_observation.servo_encoder_positions),
            (self._sensors.servo_encoder_velocities, simulator_observation.servo_encoder_velocities),
            (self._sensors.anchor_position, simulator_observation.world_link_positions[..., self._anchor_link_index, :]),
            (self._sensors.anchor_rotation, simulator_observation.world_link_rotations[..., self._anchor_link_index, :]),
        )

    @torch.no_grad()
    def reset(self, environment_indices: torch.Tensor) -> None:
        super().reset(environment_indices)
        simulator_observation = self._read_simulator_observation(environment_indices)
        self._simulator_observation[environment_indices] = simulator_observation
        for sensor, value in self._sensor_values():
            sensor.reset(value, environment_indices)
        if not self._sensors.imu.is_passthrough:
            self._sensors.imu.reset(
                self._simulator_observation.imu_rotation,
                self._simulator_observation.imu_angular_velocity,
                environment_indices,
            )

    @torch.no_grad()
    def measure_state(self) -> None:
        self._simulator_observation = self._read_simulator_observation()
        for sensor, value in self._sensor_values():
            sensor.update(value)
        if not self._sensors.imu.is_passthrough:
            self._sensors.imu.update(self._simulator_observation.imu_rotation, self._simulator_observation.imu_angular_velocity)

    @torch.no_grad()
    def _read_simulator_observation(self, environment_indices: torch.Tensor | None = None) -> SimulatorObservation:
        positions = self._robot.get_generalized_positions(environment_indices)
        velocities = self._robot.get_generalized_velocities(environment_indices)
        world_link_rotations = self._robot.get_links_rotation(environment_indices)
        world_link_angular_velocities = self._robot.get_links_angular_velocity(environment_indices)
        return SimulatorObservation(
            root_rotation=positions[..., 3 : self._robot.n_root_qs],
            root_velocity=velocities[..., :3],
            imu_rotation=world_link_rotations[..., self._anchor_link_index, :],
            imu_angular_velocity=world_link_angular_velocities[..., self._anchor_link_index, :],
            servo_encoder_positions=self._robot.servo_joints.get_encoder_positions(environment_indices),
            servo_encoder_velocities=self._robot.servo_joints.get_encoder_velocities(environment_indices),
            world_link_positions=self._robot.get_links_position(environment_indices),
            world_link_rotations=world_link_rotations,
            world_link_linear_velocities=self._robot.get_links_linear_velocity(environment_indices),
            world_link_angular_velocities=world_link_angular_velocities,
            contact_forces=self._robot.get_links_net_contact_force(environment_indices),
            batch_size=positions.shape[:-1],
        )

    def get_schema(self) -> dict[str, TensorSchema]:
        joint_count = self._robot.n_controlled_dofs
        link_count = self._key_link_indices.numel()

        component_sizes = {
            "anchor_position_error": 3,
            "anchor_rotation_error": 6,
            "root_linear_velocity": 3,
            "imu_angular_velocity": 3,
            "projected_gravity": 3,
            "servo_encoder_position_offsets": joint_count,
            "servo_encoder_velocities": joint_count,
        }
        sizes = {
            "motion_command": 2 * joint_count,
            **{f"{source}_{name}": size for source in ("measured", "simulated") for name, size in component_sizes.items()},
            "simulated_link_positions": 3 * link_count,
            "simulated_link_rotations": 6 * link_count,
            "previous_action": joint_count,
        }
        return {key: TensorSchema(shape=(size,), data_type=torch.float32) for key, size in sizes.items()}

    def observation(self, *, previous_action: torch.Tensor) -> dict[str, torch.Tensor]:
        simulator_observation = self._simulator_observation
        # Reference backlash positions and velocities are zero, so motor values are the desired encoder readings.
        servo_joints = self._robot.servo_joints
        motion_command = torch.cat(
            (
                servo_joints.select_motor_joint_values(self._reference_state.joint_dof_positions),
                servo_joints.select_motor_joint_values(self._reference_state.joint_dof_velocities),
            ),
            dim=-1,
        )

        inverse_imu_rotation = inv_quat(simulator_observation.imu_rotation)
        imu_angular_velocity = transform_by_quat(simulator_observation.imu_angular_velocity, inverse_imu_rotation)
        projected_gravity = transform_by_quat(self._world_gravity, inverse_imu_rotation)
        measured_imu_angular_velocity = imu_angular_velocity
        measured_projected_gravity = projected_gravity
        if not self._sensors.imu.is_passthrough:
            # Sensor buffers are reused; observations must remain stable until copied into rollout storage.
            measured_imu_angular_velocity = self._sensors.imu.angular_velocity.clone()
            measured_projected_gravity = self._sensors.imu.projected_gravity.clone()

        measured_components = self._observation_components(
            prefix="measured",
            anchor_position=self._sensors.anchor_position.observation,
            inverse_anchor_rotation=inv_quat(self._sensors.anchor_rotation.observation),
            root_velocity=self._sensors.root_velocity.observation,
            imu_angular_velocity=measured_imu_angular_velocity,
            projected_gravity=measured_projected_gravity,
            servo_encoder_positions=self._sensors.servo_encoder_positions.observation,
            servo_encoder_velocities=self._sensors.servo_encoder_velocities.observation,
        )

        anchor_position = simulator_observation.world_link_positions[..., self._anchor_link_index, :]
        inverse_anchor_rotation = inv_quat(simulator_observation.world_link_rotations[..., self._anchor_link_index, :])
        simulated_components = self._observation_components(
            prefix="simulated",
            anchor_position=anchor_position,
            inverse_anchor_rotation=inverse_anchor_rotation,
            root_velocity=simulator_observation.root_velocity,
            imu_angular_velocity=imu_angular_velocity,
            projected_gravity=projected_gravity,
            servo_encoder_positions=simulator_observation.servo_encoder_positions,
            servo_encoder_velocities=simulator_observation.servo_encoder_velocities,
        )

        world_link_positions = simulator_observation.world_link_positions.index_select(-2, self._key_link_indices)
        world_link_rotations = simulator_observation.world_link_rotations.index_select(-2, self._key_link_indices)
        link_positions = transform_by_quat(
            world_link_positions - anchor_position.unsqueeze(-2), inverse_anchor_rotation.unsqueeze(-2)
        ).flatten(start_dim=-2)
        link_rotations = quat_to_rot6d(transform_quat_by_quat(world_link_rotations, inverse_anchor_rotation.unsqueeze(-2))).flatten(
            start_dim=-2
        )

        return {
            "motion_command": motion_command,
            **measured_components,
            **simulated_components,
            "simulated_link_positions": link_positions,
            "simulated_link_rotations": link_rotations,
            "previous_action": previous_action.clone(),
        }

    def _observation_components(
        self,
        *,
        prefix: str,
        anchor_position: torch.Tensor,
        inverse_anchor_rotation: torch.Tensor,
        root_velocity: torch.Tensor,
        imu_angular_velocity: torch.Tensor,
        projected_gravity: torch.Tensor,
        servo_encoder_positions: torch.Tensor,
        servo_encoder_velocities: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Build named features from measured sensor readings or clean simulated values."""
        reference_anchor_position = self._reference_state.world_link_positions[..., self._anchor_link_index, :]
        reference_anchor_rotation = self._reference_state.world_link_rotations[..., self._anchor_link_index, :]
        anchor_position_error = transform_by_quat(reference_anchor_position - anchor_position, inverse_anchor_rotation)
        anchor_rotation_error = quat_to_rot6d(transform_quat_by_quat(reference_anchor_rotation, inverse_anchor_rotation))
        root_rotation = self._simulator_observation.root_rotation
        root_linear_velocity = inv_transform_by_quat(root_velocity, root_rotation)
        servo_encoder_position_offsets = servo_encoder_positions - self._robot.default_encoder_positions

        return {
            f"{prefix}_anchor_position_error": anchor_position_error,
            f"{prefix}_anchor_rotation_error": anchor_rotation_error,
            f"{prefix}_root_linear_velocity": root_linear_velocity,
            f"{prefix}_imu_angular_velocity": imu_angular_velocity,
            f"{prefix}_projected_gravity": projected_gravity,
            f"{prefix}_servo_encoder_position_offsets": servo_encoder_position_offsets,
            f"{prefix}_servo_encoder_velocities": servo_encoder_velocities.clone(),
        }

    def _compute_reward(
        self,
        *,
        current_action: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        simulator_observation = self._simulator_observation
        reference = self._reference_state
        anchor_position = simulator_observation.world_link_positions[..., self._anchor_link_index, :]
        reference_anchor_position = reference.world_link_positions[..., self._anchor_link_index, :]
        anchor_rotation = simulator_observation.world_link_rotations[..., self._anchor_link_index, :]
        reference_anchor_rotation = reference.world_link_rotations[..., self._anchor_link_index, :]

        anchor_position_error = (anchor_position - reference_anchor_position).square().sum(dim=-1)
        anchor_position_reward = torch.exp(-anchor_position_error / self._tracking_position_variance)

        anchor_rotation_difference = transform_quat_by_quat(inv_quat(reference_anchor_rotation), anchor_rotation)
        anchor_rotation_error = quat_to_rotation_vector(anchor_rotation_difference).square().sum(dim=-1)
        anchor_rotation_reward = torch.exp(-anchor_rotation_error / 0.4**2)

        # ignore heading rotation for body alignment
        alignment_rotation = inv_quat(inverse_heading_rotation(anchor_rotation_difference)).unsqueeze(-2)
        # try to match the reference height
        alignment_position = torch.cat((anchor_position[..., :2], reference_anchor_position[..., 2:3]), dim=-1).unsqueeze(-2)

        world_link_positions = simulator_observation.world_link_positions.index_select(-2, self._key_link_indices)
        world_reference_link_positions = reference.world_link_positions.index_select(-2, self._key_link_indices)
        aligned_reference_link_positions = alignment_position + transform_by_quat(
            world_reference_link_positions - reference_anchor_position.unsqueeze(-2), alignment_rotation
        )
        body_position_error = (world_link_positions - aligned_reference_link_positions).square().sum(dim=-1).mean(dim=-1)
        body_position_reward = torch.exp(-body_position_error / self._tracking_position_variance)

        world_link_rotations = simulator_observation.world_link_rotations.index_select(-2, self._key_link_indices)
        world_reference_link_rotations = reference.world_link_rotations.index_select(-2, self._key_link_indices)
        aligned_reference_link_rotations = transform_quat_by_quat(world_reference_link_rotations, alignment_rotation)
        body_rotation_difference = transform_quat_by_quat(inv_quat(aligned_reference_link_rotations), world_link_rotations)
        body_rotation_error = quat_to_rotation_vector(body_rotation_difference).square().sum(dim=-1).mean(dim=-1)
        body_rotation_reward = torch.exp(-body_rotation_error / 0.4**2)

        world_link_linear_velocities = simulator_observation.world_link_linear_velocities.index_select(-2, self._key_link_indices)
        world_reference_link_linear_velocities = reference.world_link_linear_velocities.index_select(-2, self._key_link_indices)
        body_linear_velocity_error = (
            (world_link_linear_velocities - world_reference_link_linear_velocities).square().sum(dim=-1).mean(dim=-1)
        )
        body_linear_velocity_reward = torch.exp(-body_linear_velocity_error / self._tracking_linear_velocity_variance)

        world_link_angular_velocities = simulator_observation.world_link_angular_velocities.index_select(-2, self._key_link_indices)
        world_reference_link_angular_velocities = reference.world_link_angular_velocities.index_select(-2, self._key_link_indices)
        body_angular_velocity_error = (
            (world_link_angular_velocities - world_reference_link_angular_velocities).square().sum(dim=-1).mean(dim=-1)
        )
        body_angular_velocity_reward = torch.exp(-body_angular_velocity_error / 3.14**2)

        action_change = (current_action - previous_action) / self._robot.control_action_scale
        action_change_penalty = action_change.square().sum(dim=-1)

        joint_position_limit_penalty = (
            (self._soft_encoder_position_lower_bounds - simulator_observation.servo_encoder_positions).clamp_min(0.0)
            + (simulator_observation.servo_encoder_positions - self._soft_encoder_position_upper_bounds).clamp_min(0.0)
        ).sum(dim=-1)

        penalized_contact_forces = simulator_observation.contact_forces.index_select(-2, self._penalized_contact_link_indices)
        contact_force_magnitudes = torch.linalg.vector_norm(penalized_contact_forces, dim=-1)
        undesired_contact_penalty = (contact_force_magnitudes > self._contact_force_threshold).sum(
            dim=-1, dtype=simulator_observation.servo_encoder_positions.dtype
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

    def _compute_terminal(self) -> torch.Tensor:
        simulator_observation = self._simulator_observation
        reference = self._reference_state
        anchor_height = simulator_observation.world_link_positions[..., self._anchor_link_index, 2]
        reference_anchor_height = reference.world_link_positions[..., self._anchor_link_index, 2]
        anchor_height_error = (anchor_height - reference_anchor_height).abs()

        end_effector_heights = simulator_observation.world_link_positions[..., 2].index_select(-1, self._end_effector_link_indices)
        reference_end_effector_heights = reference.world_link_positions[..., 2].index_select(-1, self._end_effector_link_indices)
        end_effector_height_errors = (end_effector_heights - reference_end_effector_heights).abs()

        anchor_rotation = simulator_observation.world_link_rotations[..., self._anchor_link_index, :]
        reference_anchor_rotation = reference.world_link_rotations[..., self._anchor_link_index, :]

        anchor_gravity_z = 2.0 * anchor_rotation[..., 1:3].square().sum(dim=-1) - 1.0
        reference_anchor_gravity_z = 2.0 * reference_anchor_rotation[..., 1:3].square().sum(dim=-1) - 1.0
        anchor_tilt_error = (anchor_gravity_z - reference_anchor_gravity_z).abs()

        return (
            (anchor_height_error > self._tracking_height_threshold)
            | (end_effector_height_errors > self._tracking_height_threshold).any(dim=-1)
            | (anchor_tilt_error > 0.8)
        )

    def evaluate_step(
        self,
        *,
        current_action: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> TaskEvaluation:
        reward, reward_components = self._compute_reward(
            current_action=current_action,
            previous_action=previous_action,
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

        terminal = self._compute_terminal()
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
