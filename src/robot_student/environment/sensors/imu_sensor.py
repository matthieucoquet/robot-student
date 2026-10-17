import torch
from genesis.utils.geom import transform_quat_by_quat, xyz_to_quat

from robot_student.environment.sensors.sensor import Sensor


class IMUSensor(Sensor):
    @torch.no_grad()
    def initialize(self, rotation: torch.Tensor, angular_velocity: torch.Tensor) -> None:
        self._imu_values = torch.cat((rotation, angular_velocity), dim=-1)
        super().initialize(self._imu_values)

    @torch.no_grad()
    def reset(
        self,
        rotation: torch.Tensor,
        angular_velocity: torch.Tensor,
        environment_indices: torch.Tensor | None = None,
    ) -> None:
        torch.cat((rotation, angular_velocity), dim=-1, out=self._imu_values)
        super().reset(self._imu_values, environment_indices)

    @torch.no_grad()
    def update(self, rotation: torch.Tensor, angular_velocity: torch.Tensor) -> None:
        torch.cat((rotation, angular_velocity), dim=-1, out=self._imu_values)
        super().update(self._imu_values)

    @property
    def rotation(self) -> torch.Tensor:
        return self.observation[..., :4]

    @property
    def angular_velocity(self) -> torch.Tensor:
        return self.observation[..., 4:]

    def _sample(self, value: torch.Tensor, output: torch.Tensor) -> None:
        rotation_noise = output[..., :3].uniform_(-self._noise_half_width, self._noise_half_width)
        output[..., :4].copy_(transform_quat_by_quat(xyz_to_quat(rotation_noise, rpy=True), value[..., :4]))
        output[..., 4:].uniform_(-self._noise_half_width, self._noise_half_width).add_(value[..., 4:])
