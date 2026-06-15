from pathlib import Path

import newton
from newton.selection import ArticulationView
from tensordict import TensorDictBase

from robot_student.engine.engine import Engine, EngineCharacter


class NewtonEngine(Engine):
    def __init__(self) -> None:
        self._builder = newton.ModelBuilder()
        self._characters = []

    def add_character(self, xml_path: Path):
        self._character_template = newton.ModelBuilder()
        self._character_template.add_mjcf(xml_path)
        self._characters.append(NewtonCharacter(xml_path.stem))

    def add_plane(self) -> None:
        self._builder.add_ground_plane()

    def build_scene(self, environment_count: int = 1, env_spacing: tuple[float, float] = (1.0, 1.0)) -> None:
        self._builder.replicate(self._character_template, world_count=environment_count)

        self._model = self._builder.finalize()

        self._solver = newton.solvers.SolverMuJoCo(self._model)

        for character in self._characters:
            character.setup_view(self._model)


class NewtonCharacter(EngineCharacter):
    def __init__(self, name: str):
        self._view_name = name
        self._view = None

    def setup_view(self, model: newton.Model):
        self._view = ArticulationView(
            model,
            self._view_name,
            exclude_joint_types=[newton.JointType.FREE],
        )

        self._joint_limit_lower = self._view.get_attribute("joint_limit_lower", model)
        self._joint_limit_upper = self._view.get_attribute("joint_limit_upper", model)

    def get_joint_limits(self) -> tuple:
        return self._joint_limit_lower, self._joint_limit_upper

    def control_pd(self, action: TensorDictBase) -> None:
        self._view.set_dof_positions(action["control"])
