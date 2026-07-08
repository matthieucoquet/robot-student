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
        self.rollout_length = rollout_length
        self.environment_count = environment_count
        self.batch_shape = torch.Size((rollout_length, environment_count))
        self.device = device

        self.observations = TensorDict(
            {
                key: torch.empty(
                    (*self.batch_shape, *tensor_schema.shape),
                    device=self.device,
                    dtype=tensor_schema.data_type,
                )
                for key, tensor_schema in schema.observations.items()
            },
            batch_size=self.batch_shape,
            device=self.device,
        )
        self.next_observations = self.observations.clone()
        self.actions = TensorDict(
            {
                key: torch.empty(
                    (*self.batch_shape, *tensor_schema.shape),
                    device=self.device,
                    dtype=tensor_schema.data_type,
                )
                for key, tensor_schema in schema.actions.items()
            },
            batch_size=self.batch_shape,
            device=self.device,
        )

        self.rewards = torch.empty(self.batch_shape, device=self.device, dtype=scalar_data_type)
        self.terminals = torch.empty(self.batch_shape, device=self.device, dtype=torch.bool)
        self.truncated = torch.empty_like(self.terminals)
        self.log_probabilities = torch.empty_like(self.rewards)
        self.advantages = torch.empty_like(self.rewards)
        self.returns = torch.empty_like(self.rewards)

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
        for key, storage in self.observations.items():
            storage[step_index].copy_(observation[key])
        for key, storage in self.next_observations.items():
            storage[step_index].copy_(next_observation[key])
        for key, storage in self.actions.items():
            storage[step_index].copy_(action[key])

        self.rewards[step_index].copy_(reward)
        self.terminals[step_index].copy_(terminal)
        self.truncated[step_index].copy_(truncated)
        self.log_probabilities[step_index].copy_(log_probability)

        self._next_step_index += 1
