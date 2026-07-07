import logging

import torch
from torch import nn
from torch.optim import Optimizer

from robot_student.algorithm.rollout_buffer import RolloutBuffer
from robot_student.environment.environment import Environment


class PPO:
    def __init__(
        self,
        environment: Environment,
        policy: nn.Module,
        value_function: nn.Module,
        policy_optimizer: Optimizer,
        value_optimizer: Optimizer,
        rollout_buffer: RolloutBuffer,
    ) -> None:
        self._environment = environment
        self._policy = policy
        self._value_function = value_function
        self._policy_optimizer = policy_optimizer
        self._value_optimizer = value_optimizer
        self._rollout_buffer = rollout_buffer
        self._logger = logging.getLogger(__name__)

    def train(self, experiment_storage, iteration_count: int, checkpoint_interval: int) -> None:
        self._policy.train()
        self._value_function.train()

        self._observations = self._environment.reset()

        for i in range(iteration_count):
            self._collect_rollouts()
            self._logger.debug("PPO iteration")

            experiment_storage.metrics.log_scalar("test", i * 0.1, i)

            if i % checkpoint_interval == 0:
                experiment_storage.checkpoint.save(
                    {
                        "policy": self._policy.state_dict(),
                        "value_function": self._value_function.state_dict(),
                        "policy_optimizer": self._policy_optimizer.state_dict(),
                        "value_optimizer": self._value_optimizer.state_dict(),
                    },
                    i,
                )

    @torch.no_grad()
    def _collect_rollouts(self) -> None:
        self._rollout_buffer.reset()
        for _ in range(self._rollout_buffer.rollout_length):
            action, log_probability = self._policy.sample_action_with_log_prob(self._observations)

            next_observations, reward, terminal, truncated = self._environment.step(action)

            self._rollout_buffer.add_transition(
                observation=self._observations,
                action=action,
                log_probability=log_probability,
                reward=reward,
                terminal=terminal,
                truncated=truncated,
                next_observation=next_observations,
            )
            done = torch.logical_or(terminal, truncated)
            self._observations = self._environment.reset_done(done)

        self._finalize_rollouts()

    def _finalize_rollouts(self) -> None:
        observations = self._rollout_buffer.observations
        next_observations = self._rollout_buffer.next_observations

        self._policy.update_normalizer(next_observations)
        self._value_function.update_normalizer(next_observations)

        next_values = self._value_function(next_observations)
        values = self._value_function(observations)
