import logging

import torch
from torch import nn
from torch.optim import Optimizer


class PPO:
    def __init__(
        self,
        policy: nn.Module,
        value_function: nn.Module,
        policy_optimizer: Optimizer,
        value_optimizer: Optimizer,
    ) -> None:
        self._policy = policy
        self._value_function = value_function
        self._policy_optimizer = policy_optimizer
        self._value_optimizer = value_optimizer
        self._rollout_buffer = None
        self._logger = logging.getLogger(__name__)

    def train(self, experiment_storage, iteration_count: int, checkpoint_interval: int) -> None:
        self._policy.train()
        self._value_function.train()

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
        # self._rollout_buffer.reset()
        pass
