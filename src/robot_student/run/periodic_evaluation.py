from dataclasses import dataclass

import torch

from robot_student.model import Policy, PolicyConfiguration
from robot_student.util.seed import isolated_random_seed

from .environment_factory import EnvironmentFactory


@dataclass(frozen=True, kw_only=True, slots=True)
class EvaluationConfiguration:
    environment_factory: EnvironmentFactory
    seed: int = 0
    maximum_steps: int = 1_000

    def __post_init__(self) -> None:
        if self.environment_factory.environment_count <= 0:
            raise ValueError("Evaluation environment_count must be positive")
        if self.maximum_steps <= 0:
            raise ValueError("Evaluation maximum_steps must be positive")


class PolicyEvaluator:
    """Evaluate one episode per environment in a persistent, independent scene."""

    def __init__(
        self,
        configuration: EvaluationConfiguration,
        *,
        policy_configuration: PolicyConfiguration,
        use_cuda: bool,
    ) -> None:
        self._configuration = configuration
        factory = configuration.environment_factory
        with isolated_random_seed(configuration.seed):
            self._engine = factory.create_engine(use_cuda=use_cuda, seed=configuration.seed)
            self._environment = factory.create_environment(engine=self._engine)
            self._policy = Policy(self._environment.schema, configuration=policy_configuration, device=self._environment.device)
            self._policy.eval()

        count = self._environment.count
        device = self._environment.device
        self._returns = torch.zeros(count, dtype=torch.float32, device=device)
        self._lengths = torch.zeros(count, dtype=torch.int64, device=device)
        self._completed = torch.zeros(count, dtype=torch.bool, device=device)
        self._timeouts = torch.zeros(count, dtype=torch.bool, device=device)

    @torch.inference_mode()
    def evaluate(self, policy: Policy) -> dict[str, int | torch.Tensor]:
        self._policy.load_state_dict(policy.state_dict())
        self._returns.zero_()
        self._lengths.zero_()
        self._completed.zero_()
        self._timeouts.zero_()

        with isolated_random_seed(self._configuration.seed):
            observation = self._environment.reset()
            for step in range(self._configuration.maximum_steps):
                action = self._policy.sample_action(observation, stochastic=False)
                _, reward, terminal, truncated, _ = self._environment.step(action)
                active = ~self._completed
                self._returns.add_(torch.where(active, reward, 0.0))
                self._lengths.add_(active)
                done = terminal | truncated
                self._timeouts.logical_or_(active & truncated)
                self._completed.logical_or_(done)

                if self._completed.all().item() or step + 1 == self._configuration.maximum_steps:
                    break
                # Completed slots may keep simulating, but only their first episode is measured.
                observation = self._environment.reset_done(done)

        self._timeouts.logical_or_(~self._completed)
        lengths = self._lengths.to(self._returns.dtype)
        mean_length = lengths.mean()
        return {
            "eval/episode_count": self._environment.count,
            "eval/episode_return_mean": self._returns.mean(),
            "eval/episode_length_mean": mean_length,
            "eval/episode_duration_seconds_mean": mean_length / self._configuration.environment_factory.control_frequency,
            "eval/reward_per_step_mean": (self._returns / lengths).mean(),
            "eval/timeout_fraction": self._timeouts.float().mean(),
        }
