from dataclasses import dataclass, fields

import genesis as gs
import torch
from genesis.engine.entities import RigidEntity
from genesis.utils.geom import transform_quat_by_quat, xyz_to_quat

from robot_student.engine.control_mode import ControlMode, PositionControlMode
from robot_student.engine.robot_state import NoiseConfiguration, RobotState

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


class Robot(KinematicRobot):
    def __init__(
        self,
        entity: RigidEntity,
        control_mode: ControlMode,
        *,
        noise_configuration: NoiseConfiguration | None = None,
        domain_randomization_configuration: DomainRandomizationConfiguration | None = None,
    ) -> None:
        super().__init__(entity)
        self._domain_randomization_configuration = domain_randomization_configuration
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
        self._setup_controlled_joints()
        self.n_controlled_dofs = len(self._controlled_dof_indices)

    @torch.no_grad()
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
            self._entity.set_COM_shift(offsets, links_idx_local=center_of_mass_link_indices)

    def sample_noisy_observation(self, state: RobotState) -> RobotState:
        if not self.noisy_observation_enabled:
            return state

        observation = state.clone(recurse=False)
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

    def _setup_controlled_joints(self) -> None:
        match self._control_mode:
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
            case PositionControlMode(joints=joint_settings):
                position_gain_values = []
                velocity_gain_values = []
                armature_values = []
                force_lower_bounds = []
                force_upper_bounds = []
                maximum_control_forces = []

                for joint in self._controlled_joints:
                    settings = joint_settings[joint.name]
                    force_lower_bound, force_upper_bound = settings.force_range
                    maximum_control_force = max(abs(force_lower_bound), abs(force_upper_bound))

                    for _ in joint.dofs_idx_local:
                        position_gain_values.append(settings.kp)
                        velocity_gain_values.append(settings.kd)
                        armature_values.append(settings.armature)
                        force_lower_bounds.append(force_lower_bound)
                        force_upper_bounds.append(force_upper_bound)
                        maximum_control_forces.append(maximum_control_force)

                self._entity.set_dofs_kp(position_gain_values, self._controlled_dof_indices)
                self._entity.set_dofs_kv(velocity_gain_values, self._controlled_dof_indices)
                self._entity.set_dofs_armature(armature_values, self._controlled_dof_indices)
                self._entity.set_dofs_force_range(force_lower_bounds, force_upper_bounds, self._controlled_dof_indices)
                maximum_control_forces_tensor = torch.tensor(
                    maximum_control_forces,
                    device=gs.device,
                    dtype=torch.float32,
                )
                position_gains = torch.tensor(position_gain_values, device=gs.device, dtype=torch.float32)
                self._control_action_scale = 0.25 * maximum_control_forces_tensor / position_gains  # BeyondMimic formula
                self._inverse_maximum_control_forces = maximum_control_forces_tensor.reciprocal_()
            case _:
                raise ValueError(f"Unsupported control mode: {self._control_mode}")

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
        return self._entity.get_dofs_limit(self._controlled_dof_indices)

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

        lower_bounds, upper_bounds = self._entity.get_dofs_limit(self._controlled_dof_indices)
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

        self._entity.control_dofs_position(self._control_targets, self._controlled_dof_indices)


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
