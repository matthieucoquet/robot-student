from dataclasses import dataclass

from robot_student.engine.kinematic_robot import RobotState


@dataclass(frozen=True, kw_only=True, slots=True)
class MotionClip:
    frequency: int
    frames: RobotState

    def __post_init__(self) -> None:
        if self.frequency <= 0:
            raise ValueError(f"frequency must be positive, got {self.frequency}")
        if len(self.frames.batch_size) != 1:
            raise ValueError(f"frames must have exactly one batch dimension, got {tuple(self.frames.batch_size)}")
        if self.frame_count == 0:
            raise ValueError("frames must not be empty")

    @property
    def frame_count(self) -> int:
        return self.frames.batch_size[0]
