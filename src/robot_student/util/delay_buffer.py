import genesis as gs
import torch
from genesis.utils.ring_buffer import TensorRingBuffer


class DelayBuffer:
    """Batched history with fixed per-environment delays; call reset before the first read."""

    def __init__(self, delay_range: tuple[int, int], environment_count: int, size: int) -> None:
        if not 0 <= delay_range[0] <= delay_range[1]:
            raise ValueError("delay_range must satisfy 0 <= minimum <= maximum")
        self._delay_range = delay_range
        self._buffer = TensorRingBuffer(self._delay_range[1] + 1, (environment_count, size), dtype=gs.tc_float)
        self._delay_steps = torch.randint(self._delay_range[0], self._delay_range[1] + 1, (environment_count,), device=gs.device)

    def reset(self, reset_value: torch.Tensor, environment_indices: torch.Tensor | None = None) -> None:
        selection = slice(None) if environment_indices is None else environment_indices
        self._buffer.buffer[:, selection, :] = reset_value

    def update(self, value: torch.Tensor) -> None:
        self._buffer.rotate()
        self._buffer.set(value)

    def get_delayed(self) -> torch.Tensor:
        return self._buffer.at(self._delay_steps, per_row=True)
