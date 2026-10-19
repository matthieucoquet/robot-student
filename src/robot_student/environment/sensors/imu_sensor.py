import math

import torch
from genesis.utils.geom import axis_angle_to_quat, inv_quat, transform_by_quat, transform_quat_by_quat

from robot_student.environment.sensors.sensor import Sensor


class IMUSensor(Sensor):
    """Convert world-frame link readings to IMU-frame observations with mounting error, noise, and delay.

    Mounting error rotates the nominal link frame into the measured IMU frame. Each environment samples
    a uniform random axis and an angle in [0, mounting_error_max_angle] radians at initialization, retained
    across episode resets. Both measurements use the same rotation before independent noise and a shared delay.
    """

    def __init__(
        self,
        *,
        mounting_error_max_angle: float = 0.0,
        angular_velocity_noise_half_width: float = 0.0,
        projected_gravity_noise_half_width: float = 0.0,
        delay_control_steps_range: tuple[int, int] | None = None,
    ) -> None:
        self._mounting_error_max_angle = mounting_error_max_angle
        self._mounting_rotation: torch.Tensor | None = None
        self._angular_velocity_noise_half_width = angular_velocity_noise_half_width
        self._projected_gravity_noise_half_width = projected_gravity_noise_half_width
        super().__init__(
            # Enable sampling when either measurement has noise; _sample applies their separate widths.
            noise_half_width=max(angular_velocity_noise_half_width, projected_gravity_noise_half_width),
            delay_control_steps_range=delay_control_steps_range,
        )
        self.is_passthrough = self.is_passthrough and mounting_error_max_angle == 0.0

    @torch.no_grad()
    def initialize(self, rotation: torch.Tensor, angular_velocity: torch.Tensor) -> None:
        self._world_gravity = rotation.new_tensor((0.0, 0.0, -1.0))
        self._imu_values = angular_velocity.new_empty((*angular_velocity.shape[:-1], 6))
        if self._mounting_error_max_angle > 0.0:
            axis = rotation.new_empty((*rotation.shape[:-1], 3)).normal_()
            angle = rotation.new_empty(rotation.shape[:-1]).uniform_(0.0, self._mounting_error_max_angle)
            self._mounting_rotation = axis_angle_to_quat(angle, axis)
        self._measure(rotation, angular_velocity)
        super().initialize(self._imu_values)

    @torch.no_grad()
    def reset(
        self,
        rotation: torch.Tensor,
        angular_velocity: torch.Tensor,
        environment_indices: torch.Tensor,
    ) -> None:
        self._measure(rotation, angular_velocity)
        super().reset(self._imu_values, environment_indices)

    @torch.no_grad()
    def update(self, rotation: torch.Tensor, angular_velocity: torch.Tensor) -> None:
        self._measure(rotation, angular_velocity)
        super().update(self._imu_values)

    def _measure(self, rotation: torch.Tensor, angular_velocity: torch.Tensor) -> None:
        inverse_rotation = inv_quat(rotation)
        if self._mounting_rotation is not None:
            # World -> nominal link frame -> measured IMU frame.
            inverse_rotation = transform_quat_by_quat(inverse_rotation, self._mounting_rotation)
        self._imu_values[..., :3].copy_(transform_by_quat(angular_velocity, inverse_rotation))
        self._imu_values[..., 3:].copy_(transform_by_quat(self._world_gravity, inverse_rotation))

    @property
    def angular_velocity(self) -> torch.Tensor:
        return self.observation[..., :3]

    @property
    def projected_gravity(self) -> torch.Tensor:
        return self.observation[..., 3:]

    def _sample(self, value: torch.Tensor, output: torch.Tensor) -> None:
        output[..., :3].uniform_(-self._angular_velocity_noise_half_width, self._angular_velocity_noise_half_width).add_(value[..., :3])
        output[..., 3:].uniform_(-self._projected_gravity_noise_half_width, self._projected_gravity_noise_half_width).add_(value[..., 3:])
