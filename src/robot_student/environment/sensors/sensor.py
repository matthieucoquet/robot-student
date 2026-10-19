import torch

from robot_student.util.delay_buffer import DelayBuffer


class Sensor:
    """Apply fresh uniform noise and fixed per-environment, per-component bias before observation delays.

    Calibration bias is sampled at initialization and retained through episode resets.
    """

    def __init__(
        self,
        *,
        bias_half_width: float = 0.0,
        noise_half_width: float = 0.0,
        delay_control_steps_range: tuple[int, int] | None = None,
    ) -> None:
        if delay_control_steps_range == (0, 0):
            delay_control_steps_range = None
        self.is_passthrough = bias_half_width == 0.0 and noise_half_width == 0.0 and delay_control_steps_range is None
        self._bias_half_width = bias_half_width
        self._bias: torch.Tensor | None = None
        self._noise_half_width = noise_half_width
        self._delay_control_steps_range = delay_control_steps_range
        self._delay_buffer: DelayBuffer | None = None
        self.observation: torch.Tensor

    def initialize(self, value: torch.Tensor) -> None:
        if self.is_passthrough:
            return

        self._bias = None
        if self._bias_half_width != 0.0:
            self._bias = torch.empty_like(value).uniform_(-self._bias_half_width, self._bias_half_width)
        self._delay_buffer = None
        if self._delay_control_steps_range is not None:
            self._delay_buffer = DelayBuffer(
                delay_range=self._delay_control_steps_range,
                environment_count=value.shape[0],
                measurement_shape=value.shape[1:],
                dtype=value.dtype,
                device=value.device,
            )
        self.observation = torch.empty_like(value)

    def update(self, value: torch.Tensor) -> None:
        if self.is_passthrough:
            self.observation = value
            return

        if self._noise_half_width == 0.0:
            self.observation.copy_(value)
        else:
            self._sample(value, self.observation)
        if self._bias is not None:
            self.observation.add_(self._bias)
        if self._delay_buffer is not None:
            self._delay_buffer.update(self.observation)
            self.observation.copy_(self._delay_buffer.get_delayed())

    def reset(self, value: torch.Tensor, environment_indices: torch.Tensor) -> None:
        if self.is_passthrough:
            self.observation = value
            return

        selected_value = value[environment_indices]
        if self._noise_half_width == 0.0:
            sample = selected_value
        else:
            sample = torch.empty_like(selected_value)
            self._sample(selected_value, sample)
        if self._bias is not None:
            sample.add_(self._bias[environment_indices])
        self.observation.index_copy_(0, environment_indices, sample)
        if self._delay_buffer is not None:
            self._delay_buffer.reset(sample, environment_indices)

    def _sample(self, value: torch.Tensor, output: torch.Tensor) -> None:
        output.uniform_(-self._noise_half_width, self._noise_half_width).add_(value)
