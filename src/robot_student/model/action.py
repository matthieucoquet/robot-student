from enum import StrEnum


class PositionTargetMode(StrEnum):
    ABSOLUTE = "absolute"
    DEFAULT_POSE_OFFSET = "default_pose_offset"


class ActionBoundEnforcement(StrEnum):
    NONE = "none"
    TANH_DISTRIBUTION = "tanh_distribution"
    BOUND_LOSS = "bound_loss"


__all__ = ["ActionBoundEnforcement", "PositionTargetMode"]
