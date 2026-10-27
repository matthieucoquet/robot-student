import logging
from dataclasses import dataclass
from pathlib import Path

import genesis as gs
import torch

from robot_student.engine.genesis_engine import GenesisEngine
from robot_student.motion.motion_clip import MotionClip
from robot_student.util.logging import configure_logging


@dataclass(frozen=True, kw_only=True, slots=True)
class MotionClipPlayer:
    robot_path: Path
    motion_path: Path
    use_cuda: bool = False
    seed: int = 0
    debug_level: int = logging.DEBUG
    viewer_camera_position: tuple[float, float, float] | None = None
    headless: bool = False
    recording_path: Path | None = None
    recording_camera_position: tuple[float, float, float] = (3.5, 0.5, 2.5)
    recording_resolution: tuple[int, int] = (1280, 720)
    recording_fps: int = 30

    @torch.inference_mode()
    def run(self) -> None:
        if self.headless and self.recording_path is None:
            raise ValueError("Headless playback requires recording_path")
        configure_logging(self.debug_level)

        motion_clip = torch.load(self.motion_path, map_location="cpu", weights_only=False)
        if not isinstance(motion_clip, MotionClip):
            raise TypeError(f"Expected a MotionClip in {self.motion_path}, got {type(motion_clip).__name__}")

        engine = GenesisEngine(
            cuda_backend=self.use_cuda,
            show_viewer=not self.headless,
            seed=self.seed,
            simulation_frequency=motion_clip.frequency,
            environment_count=1,
            viewer_camera_position=self.viewer_camera_position,
        )
        if self.recording_path is not None:
            self.recording_path.parent.mkdir(parents=True, exist_ok=True)
            engine.setup_recording(
                position=self.recording_camera_position,
                resolution=self.recording_resolution,
                environment_index=0,
                save_to_filename=self.recording_path,
                fps=self.recording_fps,
            )
        engine.add_ground_plane()
        robot = engine.add_kinematic_robot(self.robot_path)
        engine.follow_robot_for_recording(robot)
        engine.build_scene()

        frames = motion_clip.frames.to(engine.device)
        logger = logging.getLogger(__name__)
        logger.info(
            "Playing %s (%d frames at %d Hz)",
            self.motion_path,
            motion_clip.frame_count,
            motion_clip.frequency,
        )
        if self.recording_path is not None:
            logger.info("Recording one pass to %s", self.recording_path)
        else:
            logger.info("Close the viewer or press Ctrl+C to stop")

        try:
            if not self.headless:
                engine.follow_robot(robot)
            while True:
                for frame_index in range(motion_clip.frame_count):
                    if not self.headless and not engine.is_viewer_alive():
                        return
                    robot.set_state(frames[frame_index])
                    if frame_index == 0:
                        engine.reset_recording_camera()
                    try:
                        engine.step()
                    except gs.GenesisException:
                        if not self.headless and not engine.is_viewer_alive():
                            return
                        raise
                if self.recording_path is not None:
                    return
        except KeyboardInterrupt:
            logger.info("Motion playback stopped")
        finally:
            if self.recording_path is not None:
                engine.stop_recording()
