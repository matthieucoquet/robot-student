import torch
from tensordict import TensorDict, TensorDictBase
from torch import nn

from robot_student.environment.schema import EnvironmentSchema
from robot_student.model import MLP, create_distribution
from robot_student.model.normalizer import RunningNormalization


class Policy(nn.Module):
    def __init__(
        self,
        schema: EnvironmentSchema,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        self.observation_key = "proprioception"
        self.action_key = "control"

        observation_schema = schema.observations[self.observation_key]
        action_schema = schema.actions[self.action_key]

        self.normalizer = RunningNormalization(
            observation_schema.shape,
            clip=10.0,
            device=device,
            dtype=observation_schema.data_type,
        )
        self.body = MLP(input_shape=observation_schema.shape, output_shape=action_schema.shape, hidden_layers=[256], device=device)

    def forward(self, observation: TensorDictBase) -> torch.distributions.Distribution:
        normalized_observation = self.normalizer(observation[self.observation_key])
        mean = self.body(normalized_observation)
        # TODO take into account the action space bounds to compute the standard deviation, check mimickit
        return create_distribution(mean, standard_deviation=0.1)

    def sample_action(self, observation: TensorDictBase) -> TensorDictBase:
        distribution = self(observation)
        action = distribution.sample()
        return TensorDict({self.action_key: action}, batch_size=observation.batch_size, device=action.device)

    def sample_action_with_log_prob(self, observation: TensorDictBase) -> tuple[TensorDictBase, torch.Tensor]:
        distribution = self(observation)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        return TensorDict({self.action_key: action}, batch_size=observation.batch_size, device=action.device), log_prob

    def update_normalizer(self, observation: TensorDictBase) -> None:
        self.normalizer.update(observation[self.observation_key])


class ValueFunction(nn.Module):
    def __init__(
        self,
        schema: EnvironmentSchema,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        self.observation_key = "proprioception"
        self.action_key = "value"

        observation_schema = schema.observations[self.observation_key]

        self.normalizer = RunningNormalization(
            observation_schema.shape,
            clip=10.0,
            device=device,
            dtype=observation_schema.data_type,
        )
        self.body = MLP(input_shape=observation_schema.shape, output_shape=[1], hidden_layers=[256], device=device)

    def forward(self, observation: TensorDictBase) -> TensorDictBase:
        normalized_observation = self.normalizer(observation[self.observation_key])
        action = self.body(normalized_observation)

        return TensorDict({self.action_key: action}, batch_size=observation.batch_size, device=action.device)

    def update_normalizer(self, observation: TensorDictBase) -> None:
        self.normalizer.update(observation[self.observation_key])
