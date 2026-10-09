from dataclasses import dataclass, fields

import genesis as gs
import torch
from genesis.engine.entities import RigidEntity
from genesis.utils.geom import inv_transform_by_quat, transform_quat_by_quat, xyz_to_quat

from robot_student.engine.bam import Bam
from robot_student.engine.control_mode import BamControlMode, ControlMode, PositionControlMode
from robot_student.engine.robot_observation import NoiseConfiguration, RobotObservation
from robot_student.engine.robot_state import RobotState
from robot_student.util.delay_buffer import DelayBuffer

from .kinematic_robot import KinematicRobot


@dataclass(frozen=True, kw_only=True, slots=True)
class CenterOfMassRandomization:
    link_name: str
    x_range: tuple[float, float] = (0.0, 0.0)
    y_range: tuple[float, float] = (0.0, 0.0)
    z_range: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True, kw_only=True, slots=True)
class DomainRandomizationConfiguration:
    friction_ratio_range: tuple[float, float] | None = None
    center_of_mass: CenterOfMassRandomization | None = None
    default_joint_position_offset_range: tuple[float, float] | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class CommandDelayConfiguration:
    delay_physics_steps_range: tuple[int, int] = (0, 0)


@dataclass(frozen=True, kw_only=True, slots=True)
class ObservationDelayConfiguration:
    """Control-rate sensor latency, sampled once per environment and fixed across resets."""

    joint_velocity_delay_control_steps: int = 0
    imu_link_name: str | None = None
    imu_delay_control_steps_range: tuple[int, int] = (0, 0)


class Robot(KinematicRobot):
    def __init__(
        self,
        entity: RigidEntity,
        control_mode: ControlMode,
        *,
        noise_configuration: NoiseConfiguration | None = None,
        domain_randomization_configuration: DomainRandomizationConfiguration | None = None,
        command_delay_configuration: CommandDelayConfiguration | None = None,
        observation_delay_configuration: ObservationDelayConfiguration | None = None,
    ) -> None:
        super().__init__(entity)
        self._domain_randomization_configuration = domain_randomization_configuration
        self._command_delay_configuration = command_delay_configuration
        self._observation_delay_configuration = observation_delay_configuration
        self._noise_configuration = noise_configuration
        self._observation_noise: dict[str, float] = {}
        self.noisy_observation_enabled = False
        if noise_configuration is not None:
            for field in fields(noise_configuration):
                noise = getattr(noise_configuration, field.name)
                if noise is None or noise.half_width <= 0:
                    continue
                self._observation_noise[field.name] = noise.half_width
                self.noisy_observation_enabled = True
        self._control_mode = control_mode
        self._bam: Bam | None = None
        self._setup_controlled_joints()
        self.n_controlled_dofs = len(self._controlled_dof_indices)

    @torch.no_grad()
    def configure_post_build(self, environment_count: int) -> None:
        self.configure_domain_randomization(environment_count)
        self.configure_control_mode()
        self.configure_delays(environment_count)

    def configure_domain_randomization(self, environment_count: int) -> None:
        """Sample after scene building, before registering the scene's initial state."""
        configuration = self._domain_randomization_configuration
        if configuration is None:
            return

        if configuration.friction_ratio_range is not None:
            friction_ratio = torch.empty((environment_count, self.n_links), dtype=gs.tc_float, device=gs.device)
            friction_ratio.uniform_(*configuration.friction_ratio_range)
            self._entity.set_friction_ratio(friction_ratio)

        if configuration.center_of_mass is not None:
            center_of_mass = configuration.center_of_mass
            link_name = center_of_mass.link_name
            center_of_mass_link_indices = self.get_link_indices((link_name,))
            offsets = torch.empty((environment_count, 1, 3), dtype=gs.tc_float, device=gs.device)
            for axis, bounds in enumerate((center_of_mass.x_range, center_of_mass.y_range, center_of_mass.z_range)):
                offsets[..., axis].uniform_(*bounds)
            original_center_of_mass = self._entity.get_links_COM(links_idx_local=center_of_mass_link_indices)
            offsets.add_(original_center_of_mass)
            self._entity.set_links_COM(offsets, links_idx_local=center_of_mass_link_indices)

    def configure_delays(self, environment_count: int) -> None:
        self._command_delay_buffer = None
        if self._command_delay_configuration is not None:
            self._command_delay_buffer = DelayBuffer(
                delay_range=self._command_delay_configuration.delay_physics_steps_range,
                environment_count=environment_count,
                size=self.n_controlled_dofs,
            )

        self._joint_velocity_delay_buffer = None
        self._imu_delay_buffer = None
        self._imu_link_index = None
        self._observation_initialized = False
        configuration = self._observation_delay_configuration
        if configuration is not None:
            joint_velocity_delay_steps = configuration.joint_velocity_delay_control_steps
            if joint_velocity_delay_steps != 0:
                self._joint_velocity_delay_buffer = DelayBuffer(
                    delay_range=(joint_velocity_delay_steps, joint_velocity_delay_steps),
                    environment_count=environment_count,
                    size=self.n_joint_dofs,
                )
            if configuration.imu_delay_control_steps_range != (0, 0):
                if configuration.imu_link_name is None:
                    raise ValueError("imu_link_name is required when IMU delay is enabled")
                self._imu_link_index = self.get_link_indices((configuration.imu_link_name,))[0]
                self._imu_delay_buffer = DelayBuffer(
                    delay_range=configuration.imu_delay_control_steps_range,
                    environment_count=environment_count,
                    size=7,  # Link quaternion and world angular velocity.
                )
                self._imu_readings = torch.empty((environment_count, 7), dtype=gs.tc_float, device=gs.device)

    @torch.no_grad()
    def add_root_velocity(self, linear_velocity_offset: torch.Tensor, angular_velocity_offset: torch.Tensor) -> None:
        """Add world-frame velocity offsets, each shaped (environment_count, 3), in m/s and rad/s respectively."""
        if self.n_root_dofs != 6:
            raise ValueError("Root velocity offsets require a robot with a free root joint")
        root_dof_indices = (0, 1, 2, 3, 4, 5)
        root_velocities = self._entity.get_dofs_velocity(dofs_idx_local=root_dof_indices)
        root_rotation = self._entity.get_qpos(qs_idx_local=(3, 4, 5, 6))
        root_velocities[..., :3].add_(linear_velocity_offset)
        root_velocities[..., 3:].add_(inv_transform_by_quat(angular_velocity_offset, root_rotation))
        self._entity.set_dofs_velocity(root_velocities, dofs_idx_local=root_dof_indices)

    @torch.no_grad()
    def sample_noisy_observation(self, state: RobotState) -> RobotObservation:
        """Sample present measurements without reading or advancing delay history.

        Unchanged fields share clean storage and must be treated as read-only.
        """
        observation = RobotObservation(
            **{name: value.detach() if value.requires_grad else value for name, value in state.items()},
            batch_size=state.batch_size,
        )
        if self.noisy_observation_enabled:
            for field_name, half_width in self._observation_noise.items():
                value = getattr(state, field_name)
                if field_name in ("root_rotation", "world_link_rotations"):
                    angles = value.new_empty((*value.shape[:-1], 3)).uniform_(-half_width, half_width)
                    noise_rotation = xyz_to_quat(angles, rpy=True)
                    noisy_value = transform_quat_by_quat(noise_rotation, value)
                else:
                    noise = torch.empty_like(value).uniform_(-half_width, half_width)
                    noisy_value = noise.add_(value)
                setattr(observation, field_name, noisy_value)
        return observation

    @torch.no_grad()
    def observe(self, state: RobotState) -> RobotObservation:
        """Capture one sample for all environments, advancing history by one control step.

        Noise is sampled at capture time, before delaying the measurement packet.
        Seed histories with reset_observation before the first delayed observation.
        """
        if not self._observation_initialized and (self._joint_velocity_delay_buffer is not None or self._imu_delay_buffer is not None):
            raise RuntimeError("Call reset_observation with the initial state before observing delayed measurements")
        observation = self.sample_noisy_observation(state)

        if self._joint_velocity_delay_buffer is not None:
            self._joint_velocity_delay_buffer.update(observation.joint_dof_velocities)
            observation.joint_dof_velocities = self._joint_velocity_delay_buffer.get_delayed()

        if self._imu_delay_buffer is not None:
            self._imu_readings[:, :4] = observation.world_link_rotations[:, self._imu_link_index]
            self._imu_readings[:, 4:] = observation.world_link_angular_velocities[:, self._imu_link_index]
            self._imu_delay_buffer.update(self._imu_readings)
            delayed_imu = self._imu_delay_buffer.get_delayed()
            observation.world_link_rotations = observation.world_link_rotations.clone()
            observation.world_link_angular_velocities = observation.world_link_angular_velocities.clone()
            observation.world_link_rotations[:, self._imu_link_index] = delayed_imu[:, :4]
            observation.world_link_angular_velocities[:, self._imu_link_index] = delayed_imu[:, 4:]
        return observation

    @torch.no_grad()
    def reset_observation(self, state: RobotState, *, environment_indices: torch.Tensor | None = None) -> RobotObservation:
        """Seed selected histories after the final reset pose is installed, without advancing time.

        State contains only selected rows when environment_indices is provided. Every history
        slot receives the same initial noisy measurement, so no previous episode can leak through.
        """
        observation = self.sample_noisy_observation(state)
        if self._joint_velocity_delay_buffer is not None:
            self._joint_velocity_delay_buffer.reset(observation.joint_dof_velocities, environment_indices)
        if self._imu_delay_buffer is not None:
            imu_readings = torch.cat(
                (
                    observation.world_link_rotations[:, self._imu_link_index],
                    observation.world_link_angular_velocities[:, self._imu_link_index],
                ),
                dim=-1,
            )
            self._imu_delay_buffer.reset(imu_readings, environment_indices)
        if environment_indices is None:
            self._observation_initialized = True
        return observation

    def _setup_controlled_joints(self) -> None:
        match self._control_mode:
            case PositionControlMode(joints=None) | BamControlMode():
                # If joints is set to None, the actuator used in the mjcf are used
                self._controlled_joints = [
                    joint for joint in self._entity.joints if joint.n_dofs > 0 and (joint.desc.dofs_act_gain != 0).any()
                ]
                if not self._controlled_joints:
                    raise ValueError("No actuated joints found in the model; provide explicit PositionControlSettings")
            case PositionControlMode(joints=joint_settings):
                available_joint_names = {joint.name for joint in self._entity.joints if joint.n_dofs > 0}
                invalid_joint_names = joint_settings.keys() - available_joint_names
                if invalid_joint_names:
                    invalid_joint_names_text = ", ".join(sorted(invalid_joint_names))
                    raise ValueError(f"Control settings were provided for unknown or zero-DoF joints: {invalid_joint_names_text}")

                self._controlled_joints = [joint for joint in self._entity.joints if joint.n_dofs > 0 and joint.name in joint_settings]
            case _:
                raise ValueError(f"Unsupported control mode: {self._control_mode}")

        self._controlled_dof_indices = [
            degree_of_freedom_index for joint in self._controlled_joints for degree_of_freedom_index in joint.dofs_idx_local
        ]

    def configure_control_mode(self) -> None:
        match self._control_mode:
            case BamControlMode() as configuration:
                joint_positions = self._entity.get_dofs_position(self._controlled_dof_indices)
                self._bam = Bam(
                    entity=self._entity,
                    controlled_dofs_indices=self._controlled_dof_indices,
                    input_voltage=joint_positions.new_tensor(configuration.input_voltage),
                    input_voltage_drop_resistance=configuration.input_voltage_drop_resistance,
                    minimum_input_voltage=configuration.minimum_input_voltage,
                    max_current=configuration.max_current,
                )
                parameters = self._bam.actuator_parameters
                maximum_torque = parameters.kt * configuration.input_voltage * parameters.max_pwm / parameters.R
                maximum_control_forces = joint_positions.new_full((self.n_controlled_dofs,), maximum_torque)
                self._control_action_scale = joint_positions.new_full(
                    (self.n_controlled_dofs,), 0.25 * parameters.max_pwm / (parameters.kp * parameters.error_gain)
                )
                self._inverse_maximum_control_forces = maximum_control_forces.reciprocal()
                return
            case PositionControlMode(joints=None):
                pass
            case PositionControlMode(joints=joint_settings):
                position_gain_values = []
                velocity_gain_values = []
                armature_values = []
                force_lower_bounds = []
                force_upper_bounds = []

                for joint in self._controlled_joints:
                    settings = joint_settings[joint.name]
                    force_lower_bound, force_upper_bound = settings.force_range

                    for _ in joint.dofs_idx_local:
                        position_gain_values.append(settings.kp)
                        velocity_gain_values.append(settings.kd)
                        armature_values.append(settings.armature)
                        force_lower_bounds.append(force_lower_bound)
                        force_upper_bounds.append(force_upper_bound)

                self._entity.set_dofs_kp(position_gain_values, self._controlled_dof_indices)
                self._entity.set_dofs_kv(velocity_gain_values, self._controlled_dof_indices)
                self._entity.set_dofs_armature(armature_values, self._controlled_dof_indices)
                self._entity.set_dofs_force_range(force_lower_bounds, force_upper_bounds, self._controlled_dof_indices)
            case _:
                raise ValueError(f"Unsupported control mode: {self._control_mode}")

        position_gains = self._entity.get_dofs_kp(self._controlled_dof_indices)
        force_lower_bounds, force_upper_bounds = self._entity.get_dofs_force_range(self._controlled_dof_indices)
        # Controller settings are shared across environments at configuration time.
        if position_gains.ndim > 1:
            position_gains = position_gains[0]
        if force_lower_bounds.ndim > 1:
            force_lower_bounds, force_upper_bounds = force_lower_bounds[0], force_upper_bounds[0]
        maximum_control_forces = torch.maximum(force_lower_bounds.abs(), force_upper_bounds.abs())
        self._control_action_scale = 0.25 * maximum_control_forces / position_gains  # BeyondMimic formula
        self._inverse_maximum_control_forces = maximum_control_forces.reciprocal_()

    @property
    def control_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._control_lower_bounds, self._control_upper_bounds

    @property
    def control_action_scale(self) -> torch.Tensor:
        """Position scale corresponding to one quarter of the configured maximum effort."""
        return self._control_action_scale

    @property
    def default_control(self) -> torch.Tensor:
        return self._default_control_positions

    @property
    def default_joint_positions(self) -> torch.Tensor:
        return self._default_joint_positions

    def get_joint_dof_limits(self) -> tuple[torch.Tensor, torch.Tensor]:
        lower_bounds, upper_bounds = self._entity.get_dofs_limit(self._controlled_dof_indices)
        if lower_bounds.ndim > 1:  # if batch_dofs_info=True, bound tensors are (n_envs, n_dofs)
            lower_bounds = lower_bounds[0]
            upper_bounds = upper_bounds[0]
        return lower_bounds, upper_bounds

    def set_default_pose(self, default_pose: torch.Tensor) -> None:
        self._default_pose = default_pose.detach().clone()
        self.set_generalized_positions(self._default_pose, zero_velocity=True)

        controlled_positions = self._entity.get_dofs_position(dofs_idx_local=self._controlled_dof_indices)
        if controlled_positions.ndim > 1:
            controlled_positions = controlled_positions[0]
        self._default_control_positions = controlled_positions.detach().clone()

        joint_positions = self._default_pose[..., self.n_root_qs :]
        joint_position_offsets = torch.zeros_like(joint_positions)
        configuration = self._domain_randomization_configuration
        if configuration is not None and configuration.default_joint_position_offset_range is not None:
            joint_position_offsets.uniform_(*configuration.default_joint_position_offset_range)
        self._default_joint_positions = joint_positions + joint_position_offsets
        controlled_joint_indices = [index - self.n_root_dofs for index in self._controlled_dof_indices]
        self._control_position_offsets = joint_position_offsets[..., controlled_joint_indices]

        lower_bounds, upper_bounds = self.get_joint_dof_limits()
        self._control_lower_bounds, self._control_upper_bounds = scale_joint_position_limits(
            lower_bounds, upper_bounds, self._control_mode.action_limit_scale
        )
        self._control_targets = self._default_pose.new_empty((*self._default_pose.shape[:-1], self.n_controlled_dofs))

    def get_control_forces(self, environment_indices: torch.Tensor | None = None) -> torch.Tensor:
        return self._entity.get_dofs_control_force(
            self._controlled_dof_indices,
            envs_idx=environment_indices,
        )

    def get_normalized_control_forces(self, environment_indices: torch.Tensor | None = None) -> torch.Tensor:
        control_forces = self.get_control_forces(environment_indices)
        return control_forces.mul_(self._inverse_maximum_control_forces)

    def get_links_net_contact_force(self, environment_indices: torch.Tensor | None = None) -> torch.Tensor:
        return self._entity.get_links_net_contact_force(envs_idx=environment_indices)

    def apply_control(self, control: torch.Tensor) -> None:
        torch.add(control, self._control_position_offsets, out=self._control_targets)
        if self._control_mode.action_limit_scale is not None:
            torch.clamp(
                self._control_targets,
                min=self._control_lower_bounds,
                max=self._control_upper_bounds,
                out=self._control_targets,
            )

        if self._command_delay_buffer is None and isinstance(self._control_mode, PositionControlMode):
            self._entity.control_dofs_position(self._control_targets, self._controlled_dof_indices)

    @torch.no_grad()
    def update_actuator(self) -> None:
        if self._command_delay_buffer is not None:
            self._command_delay_buffer.update(self._control_targets)
            current_control = self._command_delay_buffer.get_delayed()
        else:
            current_control = self._control_targets

        if self._bam is not None:
            motor_torque = self._bam.step(current_control)
            self._entity.control_dofs_force(motor_torque, self._controlled_dof_indices)
        elif self._command_delay_buffer is not None:
            self._entity.control_dofs_position(current_control, self._controlled_dof_indices)

    @torch.no_grad()
    def reset(self, environment_indices: torch.Tensor | None = None) -> None:
        """Reset actuator and command history; seed sensors after task pose changes."""
        if self._command_delay_buffer is not None:
            self._command_delay_buffer.reset(self._default_control_positions, environment_indices)
        if self._bam is not None:
            self._bam.reset(environment_indices)


def scale_joint_position_limits(
    lower_bounds: torch.Tensor,
    upper_bounds: torch.Tensor,
    scale: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scale is None:
        # Keep finite bounds for the action schema even when target clamping is disabled.
        return lower_bounds, upper_bounds

    bound_centers = (lower_bounds + upper_bounds) * 0.5
    bound_half_ranges = (upper_bounds - lower_bounds) * (0.5 * scale)
    return bound_centers - bound_half_ranges, bound_centers + bound_half_ranges
