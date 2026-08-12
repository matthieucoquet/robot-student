from pathlib import Path
from typing import TYPE_CHECKING

from tensordict import TensorDict, TensorDictBase


from robot_student.environment.environment import Environment

if TYPE_CHECKING:
    from robot_student.engine.genesis_engine import GenesisEngine


class TrackerEnvironment(Environment):
    def __init__(
        self,
        engine: "GenesisEngine",
        xml_path: Path,
        environment_count: int,
        control_mode: ControlMode,
        control_frequency: int,
    ) -> None:
        self._count = environment_count
        self._engine = engine
        self._simulation_steps_per_control_step = engine.simulation_frequency // control_frequency
        self._engine.add_ground_plane()
        self._character = engine.add_physics_character(xml_path, control_mode=control_mode)

        device = engine.device
        # initial_pose_tensor = torch.tensor(
        #     initial_pose,
        #     dtype=torch.float32,
        #     device=device,
        # )
        # expected_shape = (self._character.n_qs,)
        # if initial_pose_tensor.shape != expected_shape:
        #     raise ValueError(f"initial_pose must have shape {expected_shape}, got {tuple(initial_pose_tensor.shape)}")

        self._engine.build_scene(environment_count=environment_count, env_spacing=(2.0, 2.0))
        # batched_initial_pose = initial_pose_tensor.expand(environment_count, -1).contiguous()
        # self._character.set_default_pose(batched_initial_pose)
        # self._engine.register_initial_pose()

        self._schema = self._compute_schema()

    def reset(self) -> TensorDictBase:
        pass
