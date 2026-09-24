from enum import StrEnum


class PositionTargetMode(StrEnum):
    ABSOLUTE = "absolute"  # Only scale using the joint limit
    DEFAULT_POSE_OFFSET = "default_pose_offset"  # Scale using the joint limit and add default pose as offset
    EFFORT_SCALED_ACTION = (
        "effort_scaled_action"  # Like BeyondMimic, use default pose as the target, and scale the action by the max torque
    )


class ActionBoundEnforcement(StrEnum):
    NONE = "none"
    TANH_DISTRIBUTION = "tanh_distribution"
    BOUND_LOSS = "bound_loss"


__all__ = ["ActionBoundEnforcement", "PositionTargetMode"]
