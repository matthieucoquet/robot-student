import logging
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.optim import Optimizer

from robot_student.algorithm.rollout_buffer import RolloutBuffer
from robot_student.environment.environment import Environment
from robot_student.model import ActionBoundEnforcement

if TYPE_CHECKING:
    from robot_student.util import Experiment


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
        clip_ratio: float = 0.2,
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
        self._clip_ratio = clip_ratio
        self._action_bound_enforcement = self._policy.action_bound_enforcement

        self._logger = logging.getLogger(__name__)

    def train(self, experiment: "Experiment", iteration_count: int, checkpoint_interval: int) -> None:
        self._policy.train()
        self._value_function.train()

        self._observations = self._environment.reset()

        for i in range(iteration_count):
            metrics = self._collect_rollouts()
            metrics |= self._update_value_function()
            metrics |= self._update_policy()
            with torch.no_grad():
                observations = self._rollout_buffer.observations
                self._policy.update_normalizer(observations)
                self._value_function.update_normalizer(observations)

            if i % self._metric_log_interval == 0:
                experiment.log_metrics(metrics, i)

            self._logger.debug(f"PPO iterations {i}")
            if i % checkpoint_interval == 0 or i == iteration_count - 1:
                experiment.save_checkpoint(
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

            next_observations, reward, terminal, truncated, transition_metrics = self._environment.step(action)

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
        return transition_metrics | {"train/mean_reward": self._rollout_buffer.rewards.mean()}

    def _finalize_rollouts(self) -> None:
        observations = self._rollout_buffer.observations
        next_observations = self._rollout_buffer.next_observations
        terminals = self._rollout_buffer.terminals
        truncated = self._rollout_buffer.truncated
        returns = self._rollout_buffer.returns
        rewards = self._rollout_buffer.rewards
        advantages = self._rollout_buffer.advantages

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

    def _update_value_function(self) -> dict[str, torch.Tensor]:
        observations = self._rollout_buffer.flat_observations
        returns = self._rollout_buffer.flat_returns

        log_loss_sum = torch.zeros((), device=returns.device)
        minibatch_count = 0

        for minibatch_indices in self._rollout_buffer.get_minibatches(self._value_batch_size, self._value_epoch_count):
            values = self._value_function(observations[minibatch_indices])
            loss = torch.nn.functional.mse_loss(values, returns[minibatch_indices])

            self._value_optimizer.zero_grad()
            loss.backward()
            self._value_optimizer.step()

            log_loss_sum += loss.detach()
            minibatch_count += 1

        return {"train/value_loss": log_loss_sum / minibatch_count}

    def _update_policy(self) -> dict[str, torch.Tensor]:
        observations = self._rollout_buffer.flat_observations
        actions = self._rollout_buffer.flat_actions
        old_log_probabilities = self._rollout_buffer.flat_log_probabilities
        advantages = self._rollout_buffer.flat_advantages

        log_loss_sum = torch.zeros((), device=observations.device)
        log_clip_fraction_sum = torch.zeros((), device=observations.device)
        action_bound_loss_enabled = self._action_bound_enforcement is ActionBoundEnforcement.BOUND_LOSS
        if action_bound_loss_enabled:
            log_action_bound_loss_sum = torch.zeros((), device=observations.device)
        minibatch_count = 0

        for minibatch_indices in self._rollout_buffer.get_minibatches(self._policy_batch_size, self._policy_epoch_count):
            log_probability, action_mean = self._policy.log_prob(observations[minibatch_indices], actions[minibatch_indices])

            ratio = torch.exp(log_probability - old_log_probabilities[minibatch_indices])
            clip_fraction = ((ratio - 1.0).abs() > self._clip_ratio).float().mean()
            unclipped_loss = -advantages[minibatch_indices] * ratio
            clipped_loss = -advantages[minibatch_indices] * torch.clamp(ratio, 1.0 - self._clip_ratio, 1.0 + self._clip_ratio)
            loss = torch.max(unclipped_loss, clipped_loss)
            if action_bound_loss_enabled:
                action_bound_loss = self._compute_action_bound_loss(action_mean)
                loss = loss + action_bound_loss
                log_action_bound_loss_sum += action_bound_loss.detach().mean()
            loss = loss.mean()

            self._policy_optimizer.zero_grad()
            loss.backward()
            self._policy_optimizer.step()

            log_loss_sum += loss.detach()
            log_clip_fraction_sum += clip_fraction.detach()
            minibatch_count += 1

        metrics = {
            "train/policy_loss": log_loss_sum / minibatch_count,
            "train/policy_clip_fraction": log_clip_fraction_sum / minibatch_count,
        }
        if action_bound_loss_enabled:
            metrics["train/action_bound_loss"] = log_action_bound_loss_sum / minibatch_count
        return metrics

    def _compute_action_bound_loss(self, action_mean: torch.Tensor) -> torch.Tensor:
        lower_bounds, upper_bounds = self._policy.action_bounds

        lower_bound_violation = torch.clamp_max(action_mean - lower_bounds, 0.0)
        upper_bound_violation = torch.clamp_min(action_mean - upper_bounds, 0.0)
        violation = torch.sum(torch.square(lower_bound_violation), dim=-1) + torch.sum(torch.square(upper_bound_violation), dim=-1)
        return violation
