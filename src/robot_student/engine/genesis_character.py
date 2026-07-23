import genesis as gs
import torch
from genesis.engine.entities import RigidEntity
from tensordict import TensorDictBase

from robot_student.engine.control_mode import ControlMode, PositionControlMode
from robot_student.environment.schema import TensorSchema


class GenesisCharacter:
    def __init__(self, character: RigidEntity, control_mode: ControlMode) -> None:
        self._character: RigidEntity = character
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
                        force_lower_bounds.append(force_lower_bound)
                        force_upper_bounds.append(force_upper_bound)
                        maximum_control_forces.append(maximum_control_force)

                self._character.set_dofs_kp(position_gain_values, self._controlled_dof_indices)
                self._character.set_dofs_kv(velocity_gain_values, self._controlled_dof_indices)
                self._character.set_dofs_force_range(force_lower_bounds, force_upper_bounds, self._controlled_dof_indices)
                self._inverse_maximum_control_forces = torch.tensor(
                    maximum_control_forces,
                    device=gs.device,
                    dtype=torch.float32,
                ).reciprocal_()
            case _:
                raise ValueError(f"Unsupported control mode: {self._control_mode}")

    def get_action_schema(self) -> TensorSchema:
        match self._control_mode:
            case PositionControlMode():
                lower_bounds, upper_bounds = self._character.get_dofs_limit(self._controlled_dof_indices)
                lower_bounds, upper_bounds = _scale_action_limits(lower_bounds, upper_bounds, self._control_mode.action_limit_scale)
            case _:
                raise ValueError(f"Unsupported control mode: {self._control_mode}")

        return TensorSchema(
            shape=(self.n_controlled_dofs,),
            data_type=torch.float32,
            bounds=(lower_bounds, upper_bounds),
        )

    def get_generalized_positions(self, environment_indices: torch.Tensor | None = None) -> torch.Tensor:
        return self._character.get_qpos(envs_idx=environment_indices)

    def get_generalized_velocities(self, environment_indices: torch.Tensor | None = None) -> torch.Tensor:
        return self._character.get_dofs_velocity(envs_idx=environment_indices)

    def get_joint_dof_positions(self, environment_indices: torch.Tensor | None = None) -> torch.Tensor:
        positions = self._character.get_dofs_position(envs_idx=environment_indices)
        return positions[..., self.n_root_dofs :]

    def set_generalized_positions(self, generalized_positions: torch.Tensor, zero_velocity: bool = False) -> None:
        self._character.set_qpos(generalized_positions, zero_velocity=zero_velocity)

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

    def apply_action(self, action: TensorDictBase) -> None:
        self._character.control_dofs_position(action["control"], self._controlled_dof_indices)


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
