from dataclasses import dataclass


@dataclass
class PositionControlSettings:
    kp: float
    kd: float
    force_range: tuple[float, float]


@dataclass
class PositionControlMode:
    joints: dict[str, PositionControlSettings]


ControlMode = PositionControlMode
