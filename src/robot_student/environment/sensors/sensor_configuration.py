from dataclasses import dataclass

from robot_student.environment.sensors.sensor import Sensor


@dataclass(frozen=True, kw_only=True, slots=True)
class SensorConfiguration:
    bias_half_width: float = 0.0
    noise_half_width: float = 0.0
    delay_control_steps_range: tuple[int, int] | None = None

    def create_sensor[SensorType: Sensor](self, sensor_type: type[SensorType]) -> SensorType:
        """Create a sensor with its own calibration bias, observation buffer, and delay history."""
        return sensor_type(
            bias_half_width=self.bias_half_width,
            noise_half_width=self.noise_half_width,
            delay_control_steps_range=self.delay_control_steps_range,
        )
