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
        discount: float = 0.99,
        td_lambda: float = 0.95,
        advantage_clip: float = 4.0,
        value_batch_size: int = 2,
        value_epoch_count: int = 2,
        policy_batch_size: int = 4,
        policy_epoch_count: int = 5,
        metric_log_interval: int = 10,
    ) -> None:
        self._environment = environment
        self._policy = policy
        self._value_function = value_function
        self._policy_optimizer = policy_optimizer
        self._value_optimizer = value_optimizer
        self._rollout_buffer = rollout_buffer
        self._discount = discount
        self._lambda = td_lambda
        self._advantage_clip = advantage_clip
        self._value_batch_size = value_batch_size
        self._value_epoch_count = value_epoch_count
        self._policy_batch_size = policy_batch_size
        self._policy_epoch_count = policy_epoch_count
        self._metric_log_interval = metric_log_interval
        self._logger = logging.getLogger(__name__)

    def train(self, experiment_storage, iteration_count: int, checkpoint_interval: int) -> None:
        self._policy.train()
        self._value_function.train()

        self._observations = self._environment.reset()

        for i in range(iteration_count):
            self._collect_rollouts()
            value_loss = self._update_value_function()
            policy_loss = self._update_policy()

            if i % self._metric_log_interval == 0:
                experiment_storage.metrics.log_scalar("value_loss", value_loss, i)
                experiment_storage.metrics.log_scalar("policy_loss", policy_loss, i)

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
        terminals = self._rollout_buffer.terminals
        truncated = self._rollout_buffer.truncated
        returns = self._rollout_buffer.returns
        rewards = self._rollout_buffer.rewards
        advantages = self._rollout_buffer.advantages

        self._policy.update_normalizer(next_observations)
        self._value_function.update_normalizer(next_observations)

        next_values = self._value_function(next_observations)
        next_values.masked_fill_(terminals, 0.0)
        values = self._value_function(observations)

        done = torch.logical_or(terminals, truncated).to(values.dtype)

        returns[-1].copy_(rewards[-1] + self._discount * next_values[-1])
        for i in reversed(range(0, self._rollout_buffer.rollout_length - 1)):
            current_lambda = self._lambda * (1.0 - done[i])
            returns[i].copy_(rewards[i] + self._discount * ((1.0 - current_lambda) * next_values[i] + current_lambda * returns[i + 1]))

        advantages.copy_(returns - values)

        advantage_std, advantage_mean = torch.std_mean(advantages, correction=0)
        advantage_std.clamp_min_(1e-5)
        advantages.sub_(advantage_mean)
        advantages.div_(advantage_std)
        advantages.clamp_(-self._advantage_clip, self._advantage_clip)

    def _update_value_function(self) -> float:
        observations = self._rollout_buffer.flat_observations
        returns = self._rollout_buffer.flat_returns

        log_loss_sum = torch.zeros((), device=returns.device)

        for minibatch_indices in self._rollout_buffer.get_minibatches(self._value_batch_size, self._value_epoch_count):
            values = self._value_function(observations[minibatch_indices])
            loss = torch.nn.functional.mse_loss(values, returns[minibatch_indices])

            self._value_optimizer.zero_grad()
            loss.backward()
            self._value_optimizer.step()

            log_loss_sum += loss.detach()

        log_loss_count = self._rollout_buffer.rollout_length / self._value_batch_size
        return (log_loss_sum / log_loss_count).item()

    def _update_policy(self) -> float:
        pass
        # observations = self._rollout_buffer.flat_observations
        # actions = self._rollout_buffer.flat_actions

        # for minibatch_indices in self._rollout_buffer.get_minibatches(self._policy_batch_size, self._policy_epoch_count):
        #     log_probability = self._policy.log_prob(observations[minibatch_indices], actions[minibatch_indices])
