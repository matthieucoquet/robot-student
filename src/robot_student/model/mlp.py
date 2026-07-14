from itertools import pairwise
from math import prod

import torch
from torch import nn

from robot_student.model.weight_initializer import DefaultWeightInitializer


class MLP(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, ...] | torch.Size,
        output_shape: tuple[int, ...] | torch.Size,
        *,
        hidden_layers: list[int],
        weight_initializer: DefaultWeightInitializer | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self._input_shape = input_shape
        self._output_shape = output_shape
        self._input_size = prod(self._input_shape)
        self._output_size = prod(self._output_shape)

        if weight_initializer is None:
            weight_initializer = DefaultWeightInitializer()

        layer_sizes = [self._input_size, *hidden_layers, self._output_size]
        layers: list[nn.Module] = []
        for layer_index, (input_size, output_size) in enumerate(pairwise(layer_sizes)):
            linear_layer = nn.Linear(input_size, output_size, device=device, dtype=dtype)
            is_head = layer_index == len(layer_sizes) - 2
            nonlinearity = "relu" if not is_head else None
            weight_initializer(linear_layer, is_head=is_head, nonlinearity=nonlinearity)
            layers.append(linear_layer)
            if not is_head:
                layers.append(nn.ReLU())

        self.network = nn.Sequential(*layers)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        input_rank = len(self._input_shape)
        batch_shape = input_tensor.shape[:-input_rank]
        flattened_input = input_tensor.reshape((*batch_shape, self._input_size))
        output = self.network(flattened_input)
        return output.reshape((*batch_shape, *self._output_shape))
