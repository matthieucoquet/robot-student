import torch
from genesis.utils.geom import axis_angle_to_quat, inv_quat, slerp, transform_quat_by_quat
from tensordict import TensorDict
from torch import nn

from robot_student.model import Policy
from robot_student.motion import MotionLibrary
from robot_student.motion.motion_clip import get_frame_interpolation
from robot_student.util.geometry import heading_angle, quat_to_rot6d


class BeyondMimicPolicy(nn.Module):
    """Sample reference joints and orientation for a BeyondMimic policy at times in seconds.

    Supply the measured anchor-to-world unit quaternion (wxyz) at playback start as
    initial_anchor_rotation on every tick, and the current reading as anchor_rotation.
    Only the starting yaw aligns the clip. The actor tracks the full aligned rotation.
    """

    computed_observation_keys = ("motion_command", "anchor_rotation_error")

    def __init__(
        self,
        policy: Policy,
        motion_library: MotionLibrary,
        motor_joint_dof_indices: torch.Tensor,
        *,
        anchor_link_index: int,
        motion_index: int = 0,
    ) -> None:
        super().__init__()
        for key in self.computed_observation_keys:
            if key not in policy.observation_keys:
                raise ValueError(f"The BeyondMimic policy requires the {key} observation")
        self.policy = policy
        first_frame = motion_library.frame_starts[motion_index].item()
        frame_count = motion_library.frame_counts[motion_index].item()
        clip_frames = slice(first_frame, first_frame + frame_count)
        frames = motion_library.frames
        motor_positions = frames.joint_dof_positions[clip_frames].index_select(-1, motor_joint_dof_indices)
        motor_velocities = frames.joint_dof_velocities[clip_frames].index_select(-1, motor_joint_dof_indices)
        self.register_buffer("motion_commands", torch.cat((motor_positions, motor_velocities), dim=-1))
        self.register_buffer("anchor_rotations", frames.world_link_rotations[clip_frames, anchor_link_index].clone())
        self.register_buffer("duration_seconds", motion_library.motion_durations[motion_index].clone())
        self.register_buffer("initial_reference_heading", heading_angle(self.anchor_rotations[0]))

    def forward(
        self,
        motion_time: torch.Tensor,
        initial_anchor_rotation: torch.Tensor,
        anchor_rotation: torch.Tensor,
        observations: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        first_frame, second_frame, blend = get_frame_interpolation(motion_time, self.duration_seconds, self.motion_commands.shape[0])
        observation = TensorDict(observations, batch_size=motion_time.shape)
        observation["motion_command"] = torch.lerp(self.motion_commands[first_frame], self.motion_commands[second_frame], blend)
        reference_anchor_rotation = slerp(self.anchor_rotations[first_frame], self.anchor_rotations[second_frame], blend)
        heading_offset = heading_angle(initial_anchor_rotation) - self.initial_reference_heading
        heading_rotation = axis_angle_to_quat(heading_offset, heading_offset.new_tensor((0.0, 0.0, 1.0)))
        aligned_reference_rotation = transform_quat_by_quat(reference_anchor_rotation, heading_rotation)
        observation["anchor_rotation_error"] = quat_to_rot6d(transform_quat_by_quat(aligned_reference_rotation, inv_quat(anchor_rotation)))
        return self.policy.sample_action(observation, stochastic=False)[self.policy.action_key]
