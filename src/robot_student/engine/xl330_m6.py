import math
from dataclasses import dataclass, field


@dataclass(frozen=True, kw_only=True, slots=True)
class Xl330ActuatorParameters:
    kt: float = 0.3459739511711113
    R: float = 2.501880848390462
    armature: float = 0.001573222677933975
    # q_offset: float = 0.01499659309559756
    # command_delay: float = 0.010220594809412403
    friction_base: float = 0.011919956825702578
    friction_stribeck: float = 0.0008513719673776765
    load_friction_motor: float = 0.22781736172050673
    load_friction_external: float = 0.10651211228971481
    load_friction_motor_stribeck: float = 1.4725366831870136e-08
    load_friction_external_stribeck: float = 0.14201808120858034
    load_friction_motor_quad: float = 0.00526632315329998
    load_friction_external_quad: float = 0.00298581585293041
    dtheta_stribeck: float = 0.2606668812858925
    alpha: float = 8.528815753151498
    friction_viscous: float = 0.005788445053875673
    kp: float = 200
    encoder_counts_per_rev: int = 4096
    kp_divisor: float = 256
    pwm_limit: int = 885
    max_pwm: float = 1.0
    max_current: float | None = None
    error_gain: float = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "error_gain", self.encoder_counts_per_rev / (2 * math.pi * self.kp_divisor * self.pwm_limit))
