import math

from torch import nn


class DefaultWeightInitializer:
    def __call__(self, layer: nn.Module, **_: object) -> None:
        return


class OrthogonalInitializer(DefaultWeightInitializer):
    def __init__(self, head_gain: float = 1.0) -> None:
        self._head_gain = head_gain

    def __call__(
        self,
        layer: nn.Module,
        *,
        is_head: bool = False,
        **_: object,
    ) -> None:
        if not isinstance(layer, nn.Linear):
            return

        gain = self._head_gain if is_head else math.sqrt(2.0)
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


class KaimingXavierInitializer(DefaultWeightInitializer):
    def __init__(self, head_scale: float | None = None) -> None:
        self._head_scale = head_scale

    def __call__(
        self,
        layer: nn.Module,
        *,
        nonlinearity: str | None = None,
        is_head: bool = False,
        **_: object,
    ) -> None:
        if not isinstance(layer, nn.Linear):
            return

        if is_head and self._head_scale is not None:
            nn.init.uniform_(layer.weight, -self._head_scale, self._head_scale)

        elif nonlinearity == "relu":
            nn.init.kaiming_normal_(
                layer.weight,
                mode="fan_in",
                nonlinearity="relu",
            )
        else:
            nn.init.xavier_normal_(layer.weight)

        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
