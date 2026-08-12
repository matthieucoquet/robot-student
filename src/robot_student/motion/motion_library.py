from pathlib import Path

import torch

from robot_student.motion.motion import Motion


class MotionLibrary:
    def __init__(self, motion_paths: list[Path]) -> None:
        self.motions: list[Motion] = []

        for motion_path in motion_paths:
            motion = torch.load(motion_path, weights_only=False)
            if not isinstance(motion, Motion):
                raise TypeError(f"Expected a Motion in {motion_path}, got {type(motion).__name__}")
            self.motions.append(motion)
