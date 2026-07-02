from pathlib import Path

import genesis as gs
import torch
from genesis.engine.entities import RigidEntity
from tensordict import TensorDictBase

from robot_student.engine.control_mode import ControlMode, PositionControlMode
from robot_student.environment.schema import TensorSchema


class GenesisEngine:
    def __init__(self, cuda_backend: bool = False, show_viewer: bool = True, seed: int | None = None) -> None:
        super().__init__()
        gs.init(backend=gs.cuda if cuda_backend else gs.cpu, seed=seed)

        self._scene = gs.Scene(show_viewer=show_viewer)
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

    def step(self, action: TensorDictBase) -> None:
        for character in self.characters:
            character.control_pd(action)
        self._scene.step()

    def reset(self) -> None:
        self._scene.reset()


class GenesisCharacter:
    def __init__(self, character: RigidEntity, control_mode: ControlMode) -> None:
        self._character: RigidEntity = character
        self._control_mode = control_mode
        self._setup_controlled_joints()
        self.n_qs = self._character.n_qs
        self.n_dofs = self._character.n_dofs
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
            case _:
                raise ValueError(f"Unsupported control mode: {self._control_mode}")

        return TensorSchema(
            shape=(self.n_controlled_dofs,),
            data_type=torch.float32,
            bounds=(lower_bounds, upper_bounds),
        )

    def get_root_position(self):
        return self._character.get_qpos()

    def get_joint_positions(self):
        return self._character.get_dofs_position()

    def control_pd(self, action: TensorDictBase) -> None:
        self._character.control_dofs_position(action["control"], self._controlled_dof_indices)
