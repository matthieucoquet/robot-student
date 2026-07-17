import torch
from tensordict import TensorDict, TensorDictBase
from torch import nn

from robot_student.environment.schema import EnvironmentSchema
from robot_student.model import MLP, ActionBoundEnforcement, create_distribution
from robot_student.model.normalizer import RunningNormalization
from robot_student.model.weight_initializer import OrthogonalInitializer


class Policy(nn.Module):
    def __init__(
        self,
        schema: EnvironmentSchema,
        device: torch.device | str | None = None,
        action_bound_enforcement: ActionBoundEnforcement = ActionBoundEnforcement.ADDITIONAL_LOSS,
    ) -> None:
        super().__init__()

        self.observation_key = "proprioception"
        self.action_key = "control"

        observation_schema = schema.observations[self.observation_key]
        action_schema = schema.actions[self.action_key]
        lower_bounds, upper_bounds = action_schema.bounds
        self.register_buffer("action_lower_bounds", lower_bounds.to(device=device, dtype=action_schema.data_type))
        self.register_buffer("action_upper_bounds", upper_bounds.to(device=device, dtype=action_schema.data_type))
        self.action_bound_enforcement = action_bound_enforcement

        self.normalizer = RunningNormalization(
            observation_schema.shape,
            clip=10.0,
            device=device,
            dtype=observation_schema.data_type,
        )
        self.body = MLP(
            input_shape=observation_schema.shape,
            output_shape=action_schema.shape,
            hidden_layers=[256, 256],
            weight_initializer=OrthogonalInitializer(head_gain=0.01),
            device=device,
        )

    @property
    def action_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.action_lower_bounds, self.action_upper_bounds

    def forward(self, observation: TensorDictBase) -> torch.Tensor:
        normalized_observation = self.normalizer(observation[self.observation_key])
        return self.body(normalized_observation)

    def create_distribution(self, mean: torch.Tensor) -> torch.distributions.Distribution:
        # TODO take into account the action space bounds to compute the standard deviation, check mimickit
        return create_distribution(
            mean,
            standard_deviation=0.1,
            action_bound_enforcement=self.action_bound_enforcement,
            bounds=self.action_bounds,
        )

    def sample_action(self, observation: TensorDictBase) -> TensorDictBase:
        mean = self(observation)
        distribution = self.create_distribution(mean)
        action = distribution.sample()
        return TensorDict({self.action_key: action}, batch_size=observation.batch_size, device=action.device)

    def sample_action_with_log_prob(self, observation: TensorDictBase) -> tuple[TensorDictBase, torch.Tensor]:
        mean = self(observation)
        distribution = self.create_distribution(mean)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        return TensorDict({self.action_key: action}, batch_size=observation.batch_size, device=action.device), log_prob

    def log_prob(self, observation: TensorDictBase, action: TensorDictBase) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self(observation)
        distribution = self.create_distribution(mean)
        return distribution.log_prob(action[self.action_key]), mean

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
        self.body = MLP(
            input_shape=observation_schema.shape,
            output_shape=(1,),
            hidden_layers=[256, 256],
            weight_initializer=OrthogonalInitializer(head_gain=1.0),
            device=device,
        )

    def forward(self, observation: TensorDictBase) -> torch.Tensor:
        normalized_observation = self.normalizer(observation[self.observation_key])
        return self.body(normalized_observation).squeeze(-1)

    def update_normalizer(self, observation: TensorDictBase) -> None:
        self.normalizer.update(observation[self.observation_key])
