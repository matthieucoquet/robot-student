import torch

from robot_student.util.delay_buffer import DelayBuffer


class Sensor:
    def __init__(
        self,
        *,
        noise_half_width: float = 0.0,
        delay_control_steps_range: tuple[int, int] | None = None,
    ) -> None:
        if delay_control_steps_range == (0, 0):
            delay_control_steps_range = None
        self.is_passthrough = noise_half_width == 0.0 and delay_control_steps_range is None
        self._noise_half_width = noise_half_width
        self._delay_control_steps_range = delay_control_steps_range
        self._delay_buffer: DelayBuffer | None = None
        self.observation: torch.Tensor

    def initialize(self, value: torch.Tensor) -> None:
        if self.is_passthrough:
            return

        self._delay_buffer = None
        if self._delay_control_steps_range is not None:
            self._delay_buffer = DelayBuffer(
                delay_range=self._delay_control_steps_range,
                environment_count=value.shape[0],
                measurement_shape=value.shape[1:],
                dtype=value.dtype,
                device=value.device,
            )

    def update(self, value: torch.Tensor) -> None:
        if self.is_passthrough:
            self.observation = value
            return

        if self._noise_half_width == 0.0:
            self.observation.copy_(value)
        else:
            self._sample(value, self.observation)
        if self._delay_buffer is not None:
            self._delay_buffer.update(self.observation)
            self.observation.copy_(self._delay_buffer.get_delayed())
        return self.observation

    def reset(self, value: torch.Tensor, environment_indices: torch.Tensor | None = None) -> None:
        if self.is_passthrough:
            self.observation = value
            return

        if environment_indices is None:
            if self._noise_half_width == 0.0:
                self.observation = value.clone()
            else:
                self.observation = torch.empty_like(value)
                self._sample(value, self.observation)
            if self._delay_buffer is not None:
                self._delay_buffer.reset(self.observation)
        else:
            selected_value = value[environment_indices]
            if self._noise_half_width == 0.0:
                sample = selected_value
            else:
                sample = torch.empty_like(selected_value)
                self._sample(selected_value, sample)
            self.observation.index_copy_(0, environment_indices, sample)
            if self._delay_buffer is not None:
                self._delay_buffer.reset(sample, environment_indices)
        return self.observation

    def _sample(self, value: torch.Tensor, output: torch.Tensor) -> None:
        output.uniform_(-self._noise_half_width, self._noise_half_width).add_(value)
