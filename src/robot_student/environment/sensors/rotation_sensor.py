import torch
from genesis.utils.geom import transform_quat_by_quat, xyz_to_quat

from robot_student.environment.sensors.sensor import Sensor


class RotationSensor(Sensor):
    """Measure wxyz quaternions with roll, pitch, and yaw noise in radians."""

    def _sample(self, value: torch.Tensor, output: torch.Tensor) -> None:
        rotation_noise = output[..., :3].uniform_(-self._noise_half_width, self._noise_half_width)
        output.copy_(transform_quat_by_quat(xyz_to_quat(rotation_noise, rpy=True), value))
