from dataclasses import dataclass


@dataclass
class PositionControlSettings:
    kp: float
    kd: float
    force_range: tuple[float, float]


@dataclass
class PositionControlMode:
    joints: dict[str, PositionControlSettings]
    action_limit_scale: float | None = 1.4


ControlMode = PositionControlMode
