from pathlib import Path

import genesis as gs
import torch
from genesis.vis.camera import Camera

from robot_student.engine.control_mode import ControlMode
from robot_student.engine.genesis_character import GenesisCharacter


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
        self._recording_camera = None
        self.characters = []

    @property
    def device(self) -> torch.device:
        return gs.device

    def add_character(self, xml_path: Path, control_mode: ControlMode) -> "GenesisCharacter":
        character = self._scene.add_entity(gs.morphs.MJCF(file=str(xml_path)))
        genesis_character = GenesisCharacter(character, control_mode=control_mode)
        self.characters.append(genesis_character)

        if self._recording_camera is not None:
            self._recording_camera.follow_entity(
                character,
                smoothing=0.05,
                fix_orientation=False,
            )

        return genesis_character

    def add_ground_plane(self) -> None:
        self._scene.add_entity(gs.morphs.Plane())

    def setup_recording(
        self,
        *,
        position: tuple[float, float, float],
        resolution: tuple[int, int] = (1280, 720),
        field_of_view: float = 50,
        far_plane: float = 100,
        environment_index: int | None = None,
        show_gui: bool = False,
        save_to_filename: Path,
        fps: int = 30,
    ) -> Camera:
        self._recording_camera = self._scene.add_camera(
            res=resolution,
            pos=position,
            fov=field_of_view,
            far=far_plane,
            env_idx=environment_index,
            GUI=show_gui,
        )
        self._scene.start_recording(
            data_func=lambda: self._recording_camera.render(rgb=True)[0],
            rec_options=gs.recorders.VideoFile(
                filename=str(save_to_filename),
                hz=fps,
                fps=fps,
            ),
        )

    def stop_recording(self):
        if not self._recording_camera:
            return
        self._scene.stop_recording()

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
