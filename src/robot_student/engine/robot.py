import genesis as gs
import torch
from genesis.engine.entities import RigidEntity

from robot_student.engine.control_mode import ControlMode, PositionControlMode
from robot_student.environment.schema import TensorSchema

from .base_robot import BaseRobot


class Robot(BaseRobot):
    def __init__(self, character: RigidEntity, control_mode: ControlMode) -> None:
        super().__init__(character)
        self._control_mode = control_mode
        self._setup_controlled_joints()
        self.n_qs = self._character.n_qs
        self.n_dofs = self._character.n_dofs
        self.n_root_dofs = self._character.links[0].n_dofs
        self.n_joint_dofs = self.n_dofs - self.n_root_dofs
        self.n_controlled_dofs = len(self._controlled_dof_indices)

    def _setup_controlled_joints(self) -> None:
        match self._control_mode:
            case PositionControlMode(joints=joint_settings):
                available_joint_names = {joint.name for joint in self._character.joints if joint.n_dofs > 0}
                invalid_joint_names = joint_settings.keys() - available_joint_names
                if invalid_joint_names:
                    invalid_joint_names_text = ", ".join(sorted(invalid_joint_names))
                    raise ValueError(f"Control settings were provided for unknown or zero-DoF joints: {invalid_joint_names_text}")

                self._controlled_joints = [joint for joint in self._character.joints if joint.n_dofs > 0 and joint.name in joint_settings]
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

                self._character.set_dofs_kp(position_gain_values, self._controlled_dof_indices)
                self._character.set_dofs_kv(velocity_gain_values, self._controlled_dof_indices)
                self._character.set_dofs_armature(armature_values, self._controlled_dof_indices)
                self._character.set_dofs_force_range(force_lower_bounds, force_upper_bounds, self._controlled_dof_indices)
                self._inverse_maximum_control_forces = torch.tensor(
                    maximum_control_forces,
                    device=gs.device,
                    dtype=torch.float32,
                ).reciprocal_()
            case _:
                raise ValueError(f"Unsupported control mode: {self._control_mode}")

    def get_action_schema(self) -> TensorSchema:
        return TensorSchema(
            shape=(self.n_controlled_dofs,),
            data_type=torch.float32,
            bounds=(self._action_lower_bounds, self._action_upper_bounds),
            default_value=self._default_control_positions,
        )

    def get_joint_dof_positions(self, environment_indices: torch.Tensor | None = None) -> torch.Tensor:
        positions = self._character.get_dofs_position(envs_idx=environment_indices)
        return positions[..., self.n_root_dofs :]

    def set_default_pose(self, default_pose: torch.Tensor) -> None:
        self._default_pose = default_pose.detach().clone()
        self.set_generalized_positions(self._default_pose, zero_velocity=True)

        controlled_positions = self._character.get_dofs_position(dofs_idx_local=self._controlled_dof_indices)
        if controlled_positions.ndim > 1:
            controlled_positions = controlled_positions[0]
        self._default_control_positions = controlled_positions.detach().clone()

        lower_bounds, upper_bounds = self._character.get_dofs_limit(self._controlled_dof_indices)
        self._action_lower_bounds, self._action_upper_bounds = _scale_action_limits(
            lower_bounds, upper_bounds, self._control_mode.action_limit_scale
        )
        self._control_targets = self._default_pose.new_empty((*self._default_pose.shape[:-1], self.n_controlled_dofs))

    def get_control_forces(self, environment_indices: torch.Tensor | None = None) -> torch.Tensor:
        return self._character.get_dofs_control_force(
            self._controlled_dof_indices,
            envs_idx=environment_indices,
        )

    def get_normalized_control_forces(self, environment_indices: torch.Tensor | None = None) -> torch.Tensor:
        control_forces = self.get_control_forces(environment_indices)
        return control_forces.mul_(self._inverse_maximum_control_forces)

    def get_root_state(self, environment_indices: torch.Tensor | None = None, relative: bool = False):
        position = self._character.get_pos(envs_idx=environment_indices, relative=relative)
        rotation = self._character.get_quat(envs_idx=environment_indices, relative=relative)
        velocity = self._character.get_vel(envs_idx=environment_indices)
        angular_velocity = self._character.get_ang(envs_idx=environment_indices)
        return position, rotation, velocity, angular_velocity

    def apply_action(self, action: torch.Tensor) -> None:
        torch.clamp(
            action,
            min=self._action_lower_bounds,
            max=self._action_upper_bounds,
            out=self._control_targets,
        )

        self._character.control_dofs_position(self._control_targets, self._controlled_dof_indices)


def _scale_action_limits(
    lower_bounds: torch.Tensor,
    upper_bounds: torch.Tensor,
    action_limit_scale: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if action_limit_scale is None:
        return lower_bounds, upper_bounds

    bound_centers = (lower_bounds + upper_bounds) * 0.5
    bound_half_ranges = (upper_bounds - lower_bounds) * (0.5 * action_limit_scale)
    return bound_centers - bound_half_ranges, bound_centers + bound_half_ranges
