from pathlib import Path

import genesis as gs
from tensordict import TensorDictBase

from robot_student.engine.engine import Engine, EngineCharacter


class GenesisEngine(Engine):
    def __init__(self, cuda_backend: bool = False, show_viewer: bool = True) -> None:
        super().__init__()
        gs.init(backend=gs.cuda if cuda_backend else gs.cpu)

        self._scene = gs.Scene(show_viewer=show_viewer)
        self.characters = []

    def add_character(self, xml_path: Path):
        character = self._scene.add_entity(gs.morphs.MJCF(file=str(xml_path)))
        genesis_character = GenesisCharacter(character)
        self.characters.append(genesis_character)
        return genesis_character

    def add_plane(self) -> None:
        self._scene.add_entity(gs.morphs.Plane())

    def build_scene(self, environment_count: int = 1, env_spacing: tuple[float, float] = (1.0, 1.0)) -> None:
        self._scene.build(n_envs=environment_count, env_spacing=env_spacing)

    def step(self, action: TensorDictBase) -> None:
        for character in self.characters:
            character.control_pd(action)
        self._scene.step()

    def reset(self) -> None:
        self._scene.reset()


class GenesisCharacter(EngineCharacter):
    def __init__(self, character):
        self._character = character
        self._controlled_dof_indices = self._compute_controlled_dof_indices()
        # self.n_qs = self._character.n_qs
        self.n_dofs = self._character.n_dofs
        self.n_controlled_dofs = len(self._controlled_dof_indices)

    def _compute_controlled_dof_indices(self) -> list[int]:
        return [
            degree_of_freedom_index
            for joint in self._character.joints
            if joint.n_dofs > 0 and bool(joint.dofs_act_gain.any())
            for degree_of_freedom_index in joint.dofs_idx_local
        ]

    def get_joint_limits(self) -> tuple:
        lower_bounds, upper_bounds = self._character.get_dofs_limit(self._controlled_dof_indices)
        return lower_bounds, upper_bounds

    def get_root_position(self):
        return self._character.get_qpos()

    def get_joint_positions(self):
        return self._character.get_dofs_position()

    def control_pd(self, action: TensorDictBase) -> None:
        self._character.control_dofs_position(action["control"], self._controlled_dof_indices)
