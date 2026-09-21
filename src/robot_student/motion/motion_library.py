from enum import StrEnum
from pathlib import Path

import torch
from genesis.utils.geom import slerp

from robot_student.engine.robot_state import RobotState
from robot_student.motion.motion_clip import MotionClip


class ReferenceSampling(StrEnum):
    ZERO = "zero"
    UNIFORM = "uniform"
    ADAPTIVE = "adaptive"


class MotionLibrary:
    def __init__(
        self,
        motion_paths: list[Path],
        device: torch.device | str,
        *,
        control_frequency: int,
        reference_sampling: ReferenceSampling = ReferenceSampling.UNIFORM,
    ) -> None:
        if not motion_paths:
            raise ValueError("At least one motion path is required")
        if control_frequency <= 0:
            raise ValueError("control_frequency must be positive")

        self._reference_sampling = ReferenceSampling(reference_sampling)
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
        if self._reference_sampling is ReferenceSampling.ADAPTIVE:
            bin_counts = [motion.frame_count // control_frequency + 1 for motion in motions]
            self._bin_counts = torch.tensor(bin_counts, dtype=torch.int64, device=device)
            self._bin_starts = self._bin_counts.cumsum(0) - self._bin_counts
            total_bin_count = sum(bin_counts)
            self._bin_motion_indices = torch.repeat_interleave(
                torch.arange(len(motions), device=device), self._bin_counts, output_size=total_bin_count
            )
            self._failure_counts = torch.zeros(total_bin_count, dtype=torch.float32, device=device)

    def sample(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        match self._reference_sampling:
            case ReferenceSampling.ZERO:
                return (
                    torch.zeros(count, dtype=torch.int64, device=self.frame_starts.device),
                    torch.zeros(count, dtype=torch.float32, device=self.frame_starts.device),
                )
            case ReferenceSampling.ADAPTIVE:
                return self._sample_adaptive(count)
            case ReferenceSampling.UNIFORM:
                pass

        sample_motion_indices = torch.multinomial(self.motion_weights, count, replacement=True)
        phase = torch.rand(count, device=self.frame_starts.device)
        durations = self.motion_durations[sample_motion_indices]
        sample_time = phase * durations
        return sample_motion_indices, sample_time

    @torch.no_grad()
    def record_failures(self, motion_indices: torch.Tensor, motion_times: torch.Tensor, failed: torch.Tensor) -> None:
        if self._reference_sampling is not ReferenceSampling.ADAPTIVE:
            return
        durations = self.motion_durations[motion_indices]
        bin_counts = self._bin_counts[motion_indices]
        local_bins = (motion_times * bin_counts / durations.clamp_min(1e-8)).long()
        local_bins = torch.minimum(local_bins, bin_counts - 1)
        bins = self._bin_starts[motion_indices] + local_bins
        self._failure_counts.mul_(0.999)
        self._failure_counts.scatter_add_(0, bins, failed.to(self._failure_counts.dtype) * 0.001)

    def _sample_adaptive(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        weights = self._failure_counts + 0.1 / self._failure_counts.numel()
        bins = torch.multinomial(weights, count, replacement=True)
        motion_indices = self._bin_motion_indices[bins]
        local_bins = bins - self._bin_starts[motion_indices]
        phase = (local_bins + torch.rand(count, device=bins.device)) / self._bin_counts[motion_indices]
        return motion_indices, phase * self.motion_durations[motion_indices]

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
            world_link_rotations=slerp(
                first_frame.world_link_rotations,
                second_frame.world_link_rotations,
                link_blend.expand_as(first_frame.world_link_rotations[..., :1]),
            ),
            world_link_linear_velocities=torch.lerp(
                first_frame.world_link_linear_velocities, second_frame.world_link_linear_velocities, link_blend
            ),
            world_link_angular_velocities=torch.lerp(
                first_frame.world_link_angular_velocities, second_frame.world_link_angular_velocities, link_blend
            ),
            batch_size=time.shape,
        )
