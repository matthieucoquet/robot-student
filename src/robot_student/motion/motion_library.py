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
            # Match BeyondMimic's frame-count / control-frequency bin construction.
            bin_counts = [motion.frame_count // control_frequency + 1 for motion in motions]
            self._motion_frequencies = torch.tensor([motion.frequency for motion in motions], dtype=torch.float32, device=device)
            self._bin_counts = torch.tensor(bin_counts, dtype=torch.int64, device=device)
            self._bin_starts = self._bin_counts.cumsum(0) - self._bin_counts
            bin_count = sum(bin_counts)
            self._bin_motion_indices = torch.repeat_interleave(
                torch.arange(len(motions), device=device), self._bin_counts, output_size=bin_count
            )
            self._bin_local_indices = torch.arange(bin_count, device=device) - self._bin_starts[self._bin_motion_indices]
            self._failure_counts = torch.zeros(bin_count, dtype=torch.float32, device=device)
            self._current_failure_counts = torch.zeros_like(self._failure_counts)

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
        """Record one control step's tracking failures; non-adaptive modes ignore feedback.

        Counts use an exponential moving average with update rate 0.001. Call once
        per control step, including steps without failures, before resetting environments.
        Motion completion is excluded here; tasks must exclude other non-failure terminations.
        """
        if self._reference_sampling is not ReferenceSampling.ADAPTIVE:
            return
        durations = self.motion_durations[motion_indices]
        bin_counts = self._bin_counts[motion_indices]
        frame_positions = motion_times * self._motion_frequencies[motion_indices]
        local_bins = (frame_positions.clamp_min(0.0) * bin_counts / self.frame_counts[motion_indices]).long()
        local_bins = torch.minimum(local_bins, bin_counts - 1)
        bins = self._bin_starts[motion_indices] + local_bins
        failures = failed & (motion_times >= 0.0) & (motion_times < durations)
        self._current_failure_counts.zero_()
        self._current_failure_counts.scatter_add_(0, bins, failures.to(self._current_failure_counts.dtype))
        self._failure_counts.lerp_(self._current_failure_counts, 0.001)

    def _sample_adaptive(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Add uniform mass before normalization, without temporal smoothing, as in BeyondMimic.

        Multiple motions share one distribution over all bins. Sampling returns frame-aligned times.
        """
        weights = self._failure_counts + 0.1 / self._failure_counts.numel()
        weights /= weights.sum()
        bins = torch.multinomial(weights, count, replacement=True)
        motion_indices = self._bin_motion_indices[bins]
        phase = (self._bin_local_indices[bins] + torch.rand(count, device=bins.device)) / self._bin_counts[motion_indices]
        frames = (phase * (self.frame_counts[motion_indices] - 1)).long()
        return motion_indices, frames / self._motion_frequencies[motion_indices]

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
