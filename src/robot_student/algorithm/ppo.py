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
            action, _log_probability = self._policy.sample_action_with_log_prob(self._observations)

            next_observations, _reward, terminal, truncated = self._environment.step(action)

            # TODO should only add to the rollout the minimum data, not value and stuff
            self._observations = next_observations
            done = torch.logical_or(terminal, truncated)
            # Store to buffer here
            self._observations = self._environment.reset_done(done)
