from pathlib import Path

import torch
from tensordict import TensorDict
from torch import nn

from robot_student.model import Policy
from robot_student.motion import MotionLibrary
from robot_student.schema import EnvironmentSchema


class OnnxTrackingModel(nn.Module):
    """Run a BeyondMimic policy with a motion command sampled in seconds."""

    def __init__(
        self,
        policy: Policy,
        motion_library: MotionLibrary,
        motor_joint_dof_indices: torch.Tensor,
        *,
        motion_index: int = 0,
    ) -> None:
        super().__init__()
        if "motion_command" not in policy.observation_keys:
            raise ValueError("The policy must use a motion_command observation")
        self.policy = policy
        self.observation_keys = tuple(key for key in policy.observation_keys if key != "motion_command")
        if not 0 <= motion_index < motion_library.frame_counts.numel():
            raise ValueError(f"Motion index {motion_index} is out of range")
        self.motion_library = motion_library
        self.motion_index = motion_index
        self.register_buffer("motor_joint_dof_indices", motor_joint_dof_indices)

    def forward(self, motion_time: torch.Tensor, observations: tuple[torch.Tensor, ...]) -> torch.Tensor:
        joint_positions, joint_velocities = self.motion_library.get_joint_state(self.motion_index, motion_time)
        observation = TensorDict(dict(zip(self.observation_keys, observations, strict=True)), batch_size=motion_time.shape)
        observation["motion_command"] = torch.cat(
            (
                joint_positions.index_select(-1, self.motor_joint_dof_indices),
                joint_velocities.index_select(-1, self.motor_joint_dof_indices),
            ),
            dim=-1,
        )
        return self.policy.sample_action(observation, stochastic=False)[self.policy.action_key]


@torch.no_grad()
def export_onnx(model: OnnxTrackingModel, schema: EnvironmentSchema, output_path: Path) -> None:
    """Export a self-contained ONNX file with a fixed batch size of one."""
    batch_size = 1
    device = model.motion_library.frame_counts.device
    motion_time = torch.zeros(batch_size, dtype=torch.float32, device=device)
    observations = tuple(
        torch.zeros((batch_size, *schema.observations[key].shape), dtype=schema.observations[key].data_type, device=device)
        for key in model.observation_keys
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (motion_time, observations),
        output_path,
        input_names=("motion_time", *model.observation_keys),
        output_names=(model.policy.action_key,),
        dynamo=True,
        external_data=False,
    )
