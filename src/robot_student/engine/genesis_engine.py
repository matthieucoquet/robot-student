from pathlib import Path

import genesis as gs
import torch

from robot_student.engine.control_mode import ControlMode
from robot_student.engine.kinematic_robot import KinematicRobot
from robot_student.engine.robot import Robot


class _MjcfCompatibleKinematicOptions(gs.options.KinematicOptions):
    # Genesis 1.3.3 passes kinematic options through MJCF's rigid-friction warnings.
    # These fields are only read while parsing and do not enable physics for kinematic entities.
    friction_cone: gs.friction_cone = gs.friction_cone.pyramidal
    enable_torsional_friction: bool = False
    enable_rolling_friction: bool = False


class GenesisEngine:
    def __init__(
        self,
        cuda_backend: bool = False,
        show_viewer: bool = True,
        seed: int | None = None,
        simulation_frequency: int = 120,
    ) -> None:
        super().__init__()

        gs.init(backend=gs.cuda if cuda_backend else gs.cpu, seed=seed)

        self.simulation_frequency = simulation_frequency
        self.time_step = 1.0 / simulation_frequency
        self._scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.time_step),
            kinematic_options=_MjcfCompatibleKinematicOptions(),
            show_viewer=show_viewer,
            profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        )
        self._recording_camera = None
        self._recording_entity = None
        self._recording_offset = None
        self.robots = []

    @property
    def device(self) -> torch.device:
        return gs.device

    def add_robot(self, xml_path: Path, control_mode: ControlMode) -> Robot:
        entity = self._scene.add_entity(gs.morphs.MJCF(file=str(xml_path)))
        robot = Robot(entity, control_mode=control_mode)
        self.robots.append(robot)

        if self._recording_camera is not None:
            self._recording_entity = entity
            self._recording_camera.follow_entity(entity, smoothing=0.2, fix_orientation=False)
            self._recording_offset = torch.as_tensor(self._recording_position, dtype=gs.tc_float, device=gs.device) - torch.as_tensor(
                entity.base_link.pos, dtype=gs.tc_float, device=gs.device
            )

        return robot

    def add_kinematic_robot(
        self,
        xml_path: Path,
        *,
        position_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        color: tuple[float, float, float, float] | None = None,
        name: str | None = None,
    ) -> KinematicRobot:
        surface = None if color is None else gs.surfaces.Default(color=color)
        entity = self._scene.add_entity(
            gs.morphs.MJCF(file=str(xml_path), collision=False, offset_pos=position_offset),
            material=gs.materials.Kinematic(),
            surface=surface,
            name=name,
        )
        return KinematicRobot(entity)

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
    ) -> None:
        self._recording_position = position
        self._recording_offset = None
        self._recording_environment_index = environment_index if environment_index is not None else 0
        self._recording_camera = self._scene.add_camera(
            res=resolution,
            pos=position,
            up=(0.0, 0.0, 1.0),
            fov=field_of_view,
            far=far_plane,
            env_idx=environment_index,
            GUI=show_gui,
        )
        self._scene.start_recording(
            data_func=self._render_recording_frame,
            rec_options=gs.recorders.VideoFile(
                filename=str(save_to_filename),
                hz=fps,
                fps=fps,
            ),
        )

    def _render_recording_frame(self):
        self._scene.visualizer.update_visual_states()
        self._recording_camera.set_pose(up=(0.0, 0.0, 1.0))
        return self._recording_camera.render(rgb=True)[0]

    def stop_recording(self):
        if not self._recording_camera:
            return
        self._scene.stop_recording()

    def build_scene(self, environment_count: int = 1, env_spacing: tuple[float, float] = (1.0, 1.0)) -> None:
        self._scene.build(n_envs=environment_count, env_spacing=env_spacing)
        for robot in self.robots:
            robot.configure_control_mode()

    def step(self) -> None:
        self._scene.step()

    def follow_robot(self, robot: KinematicRobot, smoothing: float | None = 0.05) -> None:
        viewer = self._scene.viewer
        if viewer is None:
            raise RuntimeError("Cannot follow a robot without an active viewer")
        viewer.follow_entity(robot._entity, smoothing=smoothing, fix_orientation=False)

    def is_viewer_alive(self) -> bool:
        viewer = self._scene.viewer
        return viewer is not None and viewer.is_alive()

    def reset(self, environment_indices: torch.Tensor | None = None) -> None:
        self._scene.reset(envs_idx=environment_indices)

    def reset_recording_camera(self, environment_indices: torch.Tensor | None = None) -> None:
        if self._recording_camera is None or self._recording_entity is None:
            return
        if environment_indices is None or (environment_indices == self._recording_environment_index).any():
            pelvis_position = self._recording_entity.get_pos(self._recording_environment_index, relative=False).reshape(3)
            self._recording_camera.set_pose(pos=pelvis_position + self._recording_offset, lookat=pelvis_position, up=(0.0, 0.0, 1.0))

    def register_initial_pose(self) -> None:
        self._scene.reset(state=self._scene.get_state())
