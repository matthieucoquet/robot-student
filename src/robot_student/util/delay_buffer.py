import genesis as gs
import torch
from genesis.utils.ring_buffer import TensorRingBuffer


class DelayBuffer:
    """History of ``(environment_count, *measurement_shape)`` values with fixed per-environment delays.

    Call reset before the first read.
    """

    def __init__(
        self,
        delay_range: tuple[int, int],
        environment_count: int,
        measurement_shape: tuple[int, ...],
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self._delay_range = delay_range
        shape = (environment_count, *measurement_shape)
        history_size = self._delay_range[1] + 1
        device = gs.device if device is None else device
        dtype = gs.tc_float if dtype is None else dtype
        self._buffer = TensorRingBuffer(
            history_size,
            shape,
            buffer=torch.empty((history_size, *shape), dtype=dtype, device=device),
            idx=torch.tensor(-1, dtype=torch.int64, device="cpu"),
        )
        self._delay_steps = torch.randint(self._delay_range[0], self._delay_range[1] + 1, (environment_count,), device=device)

    def reset(self, reset_value: torch.Tensor, environment_indices: torch.Tensor | None = None) -> None:
        selection = slice(None) if environment_indices is None else environment_indices
        self._buffer.buffer[:, selection] = reset_value

    def update(self, value: torch.Tensor) -> None:
        self._buffer.rotate()
        self._buffer.set(value)

    def get_delayed(self) -> torch.Tensor:
        return self._buffer.at(self._delay_steps, per_row=True)
