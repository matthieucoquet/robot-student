from dataclasses import dataclass

import torch

from robot_student.engine.robot_state import RobotState


def get_frame_interpolation(
    time: torch.Tensor, duration: torch.Tensor, frame_count: torch.Tensor | int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    last_frame = frame_count - 1
    frame_position = (time / duration).clamp(0.0, 1.0) * last_frame
    first_frame = frame_position.floor().long()
    second_frame = (first_frame + 1).clamp(max=last_frame)
    blend = (frame_position - first_frame).unsqueeze(-1)
    return first_frame, second_frame, blend


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
