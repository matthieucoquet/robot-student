from math import prod

import torch
from tensordict import TensorDict, TensorDictBase
from torch import nn

from robot_student.environment.schema import EnvironmentSchema


class MLP(nn.Module):
    def __init__(
        self,
        schema: EnvironmentSchema,
        observation_key: str,
        action_key: str,
        hidden_layer_size: int = 256,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        observation_schema = schema.observations[observation_key]
        action_schema = schema.actions[action_key]

        self.observation_key = observation_key
        self.action_key = action_key
        self.observation_shape = observation_schema.shape
        self.action_shape = action_schema.shape
        self.observation_size = prod(self.observation_shape)
        self.action_size = prod(self.action_shape)
        self.observation_data_type = observation_schema.data_type
        self.action_data_type = action_schema.data_type

        self.network = nn.Sequential(
            nn.Linear(self.observation_size, hidden_layer_size, device=device, dtype=self.observation_data_type),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, self.action_size, device=device, dtype=self.action_data_type),
        )

    def forward(self, observation: TensorDictBase) -> TensorDictBase:
        batch_shape = observation.batch_size
        flattened_observation = observation[self.observation_key].reshape((*batch_shape, self.observation_size))
        action = self.network(flattened_observation.to(dtype=self.observation_data_type))
        action = action.to(dtype=self.action_data_type).reshape((*batch_shape, *self.action_shape))

        return TensorDict({self.action_key: action}, batch_size=batch_shape, device=action.device)
