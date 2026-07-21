from pathlib import Path

import genesis as gs
import torch
from genesis.engine.entities import RigidEntity
from tensordict import TensorDictBase

from robot_student.engine.control_mode import ControlMode, PositionControlMode
from robot_student.environment.schema import TensorSchema


class GenesisEngine:
    def __init__(
        self,
        cuda_backend: bool = False,
        show_viewer: bool = True,
        seed: int | None = None,
        control_frequency: int = 100,
        simulation_frequency: int = 100,
    ) -> None:
        super().__init__()
        if simulation_frequency % control_frequency != 0:
            raise ValueError("simulation_frequency must be an integer multiple of control_frequency")

        gs.init(backend=gs.cuda if cuda_backend else gs.cpu, seed=seed)

        self.control_frequency = control_frequency
        self.simulation_frequency = simulation_frequency
        self.simulation_steps_per_control_step = simulation_frequency // control_frequency
        self._scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=1.0 / simulation_frequency),
            show_viewer=show_viewer,
            profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        )
        self.characters = []

    def add_character(self, xml_path: Path, control_mode: ControlMode) -> "GenesisCharacter":
        character = self._scene.add_entity(gs.morphs.MJCF(file=str(xml_path)))
        genesis_character = GenesisCharacter(character, control_mode=control_mode)
        self.characters.append(genesis_character)
        return genesis_character

    def add_ground_plane(self) -> None:
        self._scene.add_entity(gs.morphs.Plane())

    def build_scene(self, environment_count: int = 1, env_spacing: tuple[float, float] = (1.0, 1.0)) -> None:
        self._scene.build(n_envs=environment_count, env_spacing=env_spacing)
        for character in self.characters:
            character.configure_control_mode()

    def step(self) -> None:
        self._scene.step()

    def reset(self, environment_indices: torch.Tensor | None = None) -> None:
        self._scene.reset(envs_idx=environment_indices)

    def register_initial_pose(self) -> None:
        self._scene.reset(state=self._scene.get_state())


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
        self._controlled_joints = [joint for joint in self._character.joints if joint.n_dofs > 0 and bool(joint.dofs_act_gain.any())]
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

                for joint in self._controlled_joints:
                    settings = joint_settings[joint.name]
                    for _ in joint.dofs_idx_local:
                        position_gain_values.append(settings.kp)
                        velocity_gain_values.append(settings.kd)
                        force_lower_bound, force_upper_bound = settings.force_range
                        force_lower_bounds.append(force_lower_bound)
                        force_upper_bounds.append(force_upper_bound)

                self._character.set_dofs_kp(position_gain_values, self._controlled_dof_indices)
                self._character.set_dofs_kv(velocity_gain_values, self._controlled_dof_indices)
                self._character.set_dofs_force_range(force_lower_bounds, force_upper_bounds, self._controlled_dof_indices)
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
