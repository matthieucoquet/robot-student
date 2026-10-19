from dataclasses import dataclass


@dataclass
class PositionControlSettings:
    kp: float
    kd: float
    armature: float
    force_range: tuple[float, float]


@dataclass
class PositionControlMode:
    joints: dict[str, PositionControlSettings] | None
    action_limit_scale: float | None = 1.4  # None disables target clamping


@dataclass
class BamControlMode:
    input_voltage: float
    input_voltage_range: tuple[float, float] | None = None  # Volts.
    input_voltage_drop_resistance: float | None = None
    input_voltage_drop_resistance_range: tuple[float, float] | None = None  # Ohms.
    minimum_input_voltage: float = 0.0
    action_limit_scale: float | None = 1.4
    max_current: float | None = None
    friction_scale_range: tuple[float, float] | None = None  # Non-viscous friction multiplier; None uses 1.0.


ControlMode = PositionControlMode | BamControlMode
