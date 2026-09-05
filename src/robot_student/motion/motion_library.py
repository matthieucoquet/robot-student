from pathlib import Path

import torch
from genesis.utils.geom import slerp

from robot_student.engine.kinematic_robot import RobotState
from robot_student.motion.motion_clip import MotionClip


class MotionLibrary:
    def __init__(self, motion_paths: list[Path], device: torch.device | str) -> None:
        if not motion_paths:
            raise ValueError("At least one motion path is required")

        motions: list[MotionClip] = []

        for motion_path in motion_paths:
            motion_clip = torch.load(motion_path, map_location=device, weights_only=False)
            if not isinstance(motion_clip, MotionClip):
                raise TypeError(f"Expected a MotionClip in {motion_path}, got {type(motion_clip).__name__}")
            motions.append(motion_clip)

        self.frame_counts = torch.tensor([motion_clip.frame_count for motion_clip in motions], dtype=torch.int64, device=device)
        self.frame_starts = self.frame_counts.cumsum(dim=0) - self.frame_counts
        self.motion_weights = self.frame_counts.to(torch.float32)  # Weighting by motion length for now
        self.motion_durations = torch.tensor(
            [(motion_clip.frame_count - 1) / motion_clip.frequency for motion_clip in motions], dtype=torch.float32, device=device
        )
        self.frames = torch.cat([motion_clip.frames for motion_clip in motions], dim=0)

    def sample(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample_motion_indices = torch.multinomial(self.motion_weights, count, replacement=True)
        phase = torch.rand(count, device=self.frame_starts.device)
        durations = self.motion_durations[sample_motion_indices]
        sample_time = phase * durations
        return sample_motion_indices, sample_time

    def get_state(self, motion_indices: torch.Tensor, time: torch.Tensor) -> RobotState:
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
        link_blend = blend.unsqueeze(-2)

        return RobotState(
            root_position=torch.lerp(first_frame.root_position, second_frame.root_position, blend),
            root_rotation=slerp(first_frame.root_rotation, second_frame.root_rotation, blend),
            joint_dof_positions=torch.lerp(first_frame.joint_dof_positions, second_frame.joint_dof_positions, blend),
            root_velocity=torch.lerp(first_frame.root_velocity, second_frame.root_velocity, blend),
            root_angular_velocity=torch.lerp(first_frame.root_angular_velocity, second_frame.root_angular_velocity, blend),
            joint_dof_velocities=torch.lerp(first_frame.joint_dof_velocities, second_frame.joint_dof_velocities, blend),
            world_link_positions=torch.lerp(first_frame.world_link_positions, second_frame.world_link_positions, link_blend),
            # link_rotations=slerp(first_frame.link_rotations, second_frame.link_rotations, link_blend),
            batch_size=time.shape,
        )
