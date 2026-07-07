from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict, TensorDictBase

if TYPE_CHECKING:
    from robot_student.environment.schema import EnvironmentSchema


class RolloutBuffer:
    def __init__(
        self,
        schema: EnvironmentSchema,
        rollout_length: int,
        environment_count: int,
        device: torch.device | str | None = None,
        scalar_data_type: torch.dtype = torch.float32,
    ) -> None:
        self._rollout_length = rollout_length
        self._environment_count = environment_count
        self._batch_shape = torch.Size((rollout_length, environment_count))
        self._device = device

        self._observations = TensorDict(
            {
                key: torch.empty(
                    (*self._batch_shape, *tensor_schema.shape),
                    device=self._device,
                    dtype=tensor_schema.data_type,
                )
                for key, tensor_schema in schema.observations.items()
            },
            batch_size=self._batch_shape,
            device=self._device,
        )
        self._next_observations = self._observations.clone()
        self._actions = TensorDict(
            {
                key: torch.empty(
                    (*self._batch_shape, *tensor_schema.shape),
                    device=self._device,
                    dtype=tensor_schema.data_type,
                )
                for key, tensor_schema in schema.actions.items()
            },
            batch_size=self._batch_shape,
            device=self._device,
        )

        self._rewards = torch.empty(self._batch_shape, device=self._device, dtype=scalar_data_type)
        self._terminals = torch.empty(self._batch_shape, device=self._device, dtype=torch.bool)
        self._truncated = self._terminals.clone()
        # self._values = self._rewards.clone()
        self._log_probabilities = self._rewards.clone()
        # self._advantages = torch.empty(self._batch_shape, device=self._device, dtype=scalar_data_type)
        # self._returns = torch.empty(self._batch_shape, device=self._device, dtype=scalar_data_type)

        self._next_step_index = 0

    def reset(self) -> None:
        self._next_step_index = 0

    @torch.no_grad()
    def add_transition(
        self,
        *,
        observation: TensorDictBase,
        action: TensorDictBase,
        log_probability: torch.Tensor,
        reward: torch.Tensor,
        terminal: torch.Tensor,
        truncated: torch.Tensor,
        next_observation: TensorDictBase,
    ) -> None:
        step_index = self._next_step_index
        for key, storage in self._observations.items():
            storage[step_index].copy_(observation[key])
        for key, storage in self._next_observations.items():
            storage[step_index].copy_(next_observation[key])
        for key, storage in self._actions.items():
            storage[step_index].copy_(action[key])

        self.rewards[step_index].copy_(reward)
        self.terminals[step_index].copy_(terminal)
        self.truncated[step_index].copy_(truncated)
        self.log_probabilities[step_index].copy_(log_probability)

        self._next_step_index += 1
