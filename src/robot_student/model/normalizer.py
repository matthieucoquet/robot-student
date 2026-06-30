import torch
from torch import nn


class RunningNormalization(nn.Module):
    def __init__(
        self,
        shape: tuple[int, ...] | torch.Size,
        min_std: float = 1e-4,
        clip: float | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self._clip = clip
        self._min_var = min_std * min_std

        self.register_buffer("_count", torch.tensor(0, device=device, dtype=torch.long))
        self.register_buffer("_mean", torch.zeros(shape, device=device, dtype=dtype))
        self.register_buffer("_var", torch.ones(shape, device=device, dtype=dtype))
        self.register_buffer("_std", torch.ones(shape, device=device, dtype=dtype))

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        count_added = x.shape[0]
        var_added, mean_added = torch.var_mean(x, dim=0, correction=0)

        total_count = self._count + count_added
        rate = count_added / total_count

        mean_delta = mean_added - self._mean
        self._count.copy_(total_count)
        self._mean += rate * mean_delta
        self._var += rate * (var_added - self._var + mean_delta * (mean_added - self._mean))
        self._std.copy_(self._var).clamp_(min=self._min_var).sqrt_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self._mean) / self._std
        if self._clip is not None:
            x = torch.clamp(x, -self._clip, self._clip)
        return x
