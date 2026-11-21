import json

import torch

from .beyond_mimic_policy import BeyondMimicPolicy

CONTRACT_METADATA_KEY = "robot_student.policy_contract"


def tracking_contract_json(model: BeyondMimicPolicy, *, initial_previous_action: torch.Tensor) -> str:
    """Serialize clip timing, the first-frame pose, and the reset action in servo order."""
    clip_start_joint_positions, _ = model.motion_commands[0].chunk(2)
    contract = {
        "duration_seconds": model.duration_seconds.item(),
        "clip_start_joint_positions_rad": clip_start_joint_positions.tolist(),
        "initial_previous_action_rad": initial_previous_action.tolist(),
    }
    return json.dumps(contract, indent=2, allow_nan=False) + "\n"
