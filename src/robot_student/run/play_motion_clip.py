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

    @torch.inference_mode()
    def run(self) -> None:
        configure_logging(self.debug_level)

        motion_clip = torch.load(self.motion_path, map_location="cpu", weights_only=False)
        if not isinstance(motion_clip, MotionClip):
            raise TypeError(f"Expected a MotionClip in {self.motion_path}, got {type(motion_clip).__name__}")

        engine = GenesisEngine(
            cuda_backend=self.use_cuda,
            show_viewer=True,
            seed=self.seed,
            simulation_frequency=motion_clip.frequency,
        )
        engine.add_ground_plane()
        robot = engine.add_kinematic_robot(self.robot_path)
        engine.build_scene(environment_count=1)
        engine.follow_robot(robot)

        frames = motion_clip.frames.to(engine.device)
        logger = logging.getLogger(__name__)
        logger.info(
            "Playing %s (%d frames at %d Hz); close the viewer or press Ctrl+C to stop",
            self.motion_path,
            motion_clip.frame_count,
            motion_clip.frequency,
        )

        try:
            while engine.is_viewer_alive():
                for frame_index in range(motion_clip.frame_count):
                    robot.set_state(frames[frame_index])
                    try:
                        engine.step()
                    except gs.GenesisException:
                        if not engine.is_viewer_alive():
                            return
                        raise
        except KeyboardInterrupt:
            logger.info("Motion playback stopped")
