import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch.optim import Optimizer

from robot_student.environment.environment import Environment
from robot_student.model import Policy, PolicyConfiguration, ValueFunction, ValueFunctionConfiguration
from robot_student.model.action import ActionBoundEnforcement

from .rollout_buffer import RolloutBuffer

OptimizerFactory = Callable[[Iterable[torch.Tensor]], Optimizer]

if TYPE_CHECKING:
    pass


@dataclass(frozen=True, kw_only=True, slots=True)
class PPOConfiguration:
    policy: PolicyConfiguration
    value_function: ValueFunctionConfiguration
    policy_optimizer: OptimizerFactory
    value_function_optimizer: OptimizerFactory
    compile_models: bool = False
    rollout_length: int = 32
    discount: float = 0.99
    td_lambda: float = 0.95
    advantage_clip: float = 4.0
    value_batch_size: int = 2
    value_epoch_count: int = 2
    policy_batch_size: int = 4
    policy_epoch_count: int = 5
    clip_ratio: float = 0.2


@dataclass(frozen=True, kw_only=True, slots=True)
class PPOFactory:
    configuration: PPOConfiguration

    def create(self, *, environment: Environment):
        return PPO(environment=environment, configuration=self.configuration)


class PPO:
    def __init__(
        self,
        environment: Environment,
        configuration: PPOConfiguration,
    ) -> None:
        self._environment = environment
        device = environment.device
        self._policy = Policy(environment.schema, configuration=configuration.policy, device=device)
        self._value_function = ValueFunction(environment.schema, configuration=configuration.value_function, device=device)

        if configuration.compile_models:
            self._policy.compile()
            self._value_function.compile()

        self._policy_optimizer = configuration.policy_optimizer(self._policy.parameters())
        self._value_optimizer = configuration.value_function_optimizer(self._value_function.parameters())

        self._rollout_buffer = RolloutBuffer(
            schema=environment.schema, rollout_length=configuration.rollout_length, environment_count=environment.count, device=device
        )

        self._discount = configuration.discount
        self._lambda = configuration.td_lambda
        self._advantage_clip = configuration.advantage_clip
        self._value_batch_size = configuration.value_batch_size
        self._value_epoch_count = configuration.value_epoch_count
        self._policy_batch_size = configuration.policy_batch_size
        self._policy_epoch_count = configuration.policy_epoch_count
        self._clip_ratio = configuration.clip_ratio
        self._action_bound_enforcement = self._policy.action_bound_enforcement

        self._logger = logging.getLogger(__name__)

    @property
    def policy(self) -> Policy:
        return self._policy

    def train(self) -> None:
        self._policy.train()
        self._value_function.train()

        self._observations = self._environment.reset()

    def update(self) -> dict[str, torch.Tensor]:
        with torch.profiler.record_function("ppo.collect_rollouts"):
            metrics = self._collect_rollouts()
        with torch.profiler.record_function("ppo.update_value_function"):
            metrics |= self._update_value_function()
        with torch.profiler.record_function("ppo.update_policy"):
            metrics |= self._update_policy()
        with torch.profiler.record_function("ppo.update_normalizers"), torch.no_grad():
            observations = self._rollout_buffer.observations
            self._policy.update_normalizer(observations)
            self._value_function.update_normalizer(observations)

        return metrics

    def checkpoint(self):
        return {
            "policy": self._policy.state_dict(),
            "value_function": self._value_function.state_dict(),
            "policy_optimizer": self._policy_optimizer.state_dict(),
            "value_optimizer": self._value_optimizer.state_dict(),
        }

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

        with torch.profiler.record_function("ppo.compute_returns"):
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
        log_approximate_kl_sum = torch.zeros((), device=observations.device)
        action_bound_loss_enabled = self._action_bound_enforcement is ActionBoundEnforcement.BOUND_LOSS
        if action_bound_loss_enabled:
            log_action_bound_loss_sum = torch.zeros((), device=observations.device)
        minibatch_count = 0

        for minibatch_indices in self._rollout_buffer.get_minibatches(self._policy_batch_size, self._policy_epoch_count):
            log_probability, action_mean = self._policy.log_prob(observations[minibatch_indices], actions[minibatch_indices])

            log_ratio = log_probability - old_log_probabilities[minibatch_indices]
            ratio = torch.exp(log_ratio)
            clip_fraction = ((ratio - 1.0).abs() > self._clip_ratio).float().mean()
            approximate_kl = ((ratio - 1.0) - log_ratio).mean()
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
            log_approximate_kl_sum += approximate_kl.detach()
            minibatch_count += 1

        metrics = {
            "train/policy_loss": log_loss_sum / minibatch_count,
            "train/policy_clip_fraction": log_clip_fraction_sum / minibatch_count,
            "train/policy_approximate_kl": log_approximate_kl_sum / minibatch_count,
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
