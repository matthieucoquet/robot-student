from pathlib import Path

import torch
from genesis.utils.geom import slerp

from robot_student.motion.motion import Motion, MotionState


class MotionLibrary:
    def __init__(self, motion_paths: list[Path]) -> None:
        if not motion_paths:
            raise ValueError("At least one motion path is required")

        motions: list[Motion] = []

        for motion_path in motion_paths:
            motion = torch.load(motion_path, weights_only=False)
            if not isinstance(motion, Motion):
                raise TypeError(f"Expected a Motion in {motion_path}, got {type(motion).__name__}")
            motions.append(motion)

        frame_device = motions[0].root_position.device
        self.frame_counts = torch.tensor([motion.frame_count for motion in motions], dtype=torch.int64, device=frame_device)
        self.frame_starts = self.frame_counts.cumsum(dim=0) - self.frame_counts
        self.motion_weights = self.frame_counts.to(torch.float32)  # Weighting by motion length for now
        self.motion_durations = torch.tensor(
            [motion.frame_count / motion.frequency for motion in motions], dtype=torch.float32, device=frame_device
        )
        self.frames = torch.cat(motions, dim=0)

    def sample(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample_motion_indices = torch.multinomial(self.motion_weights, count, replacement=True)
        phase = torch.rand(count, device=self.frame_starts.device)
        durations = self.motion_durations[sample_motion_indices]
        sample_time = phase * durations
        return sample_motion_indices, sample_time

    def get_state(self, motion_indices: torch.Tensor, time: torch.Tensor) -> MotionState:
        durations = self.motion_durations[motion_indices]
        frame_counts = self.frame_counts[motion_indices]
        phase = torch.clip(time / durations, 0.0, 1.0)

        frame_start = self.frame_starts[motion_indices]
        frame_position = phase * (frame_counts - 1)

        first_frame_id = torch.floor(frame_position).to(torch.int64)
        second_frame_id = torch.min(first_frame_id + 1, frame_counts - 1)
        blend = frame_position - first_frame_id

        first_frame_id += frame_start
        second_frame_id += frame_start

        first_frame = self.frames[first_frame_id]
        second_frame = self.frames[second_frame_id]

        blend = blend.unsqueeze(-1)
        link_blend = blend.unsqueeze(-2).expand(*first_frame.link_rotations.shape[:-1], 1)

        return MotionState(
            root_position=torch.lerp(first_frame.root_position, second_frame.root_position, blend),
            root_rotation=slerp(first_frame.root_rotation, second_frame.root_rotation, blend),
            joint_dof_position=torch.lerp(first_frame.joint_dof_position, second_frame.joint_dof_position, blend),
            root_velocity=torch.lerp(first_frame.root_velocity, second_frame.root_velocity, blend),
            root_angular_velocity=torch.lerp(first_frame.root_angular_velocity, second_frame.root_angular_velocity, blend),
            joint_dof_velocities=torch.lerp(first_frame.joint_dof_velocities, second_frame.joint_dof_velocities, blend),
            link_positions=torch.lerp(first_frame.link_positions, second_frame.link_positions, link_blend),
            link_rotations=slerp(first_frame.link_rotations, second_frame.link_rotations, link_blend),
        )
