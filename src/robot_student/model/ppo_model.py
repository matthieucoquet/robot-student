from collections.abc import Callable
from dataclasses import dataclass
from math import prod

import torch
from tensordict import TensorDict, TensorDictBase
from torch import nn

from robot_student.environment.schema import EnvironmentSchema
from robot_student.model.action import ActionBoundEnforcement, PositionTargetMode
from robot_student.model.distribution import ActionDistribution
from robot_student.model.normalizer import RunningNormalization

BodyFactory = Callable[..., nn.Module]


def _get_observation_input_specification(
    schema: EnvironmentSchema,
    observation_keys: tuple[str, ...],
) -> tuple[tuple[int], torch.dtype]:
    observation_schemas = tuple(schema.observations[key] for key in observation_keys)

    data_type = observation_schemas[0].data_type
    if any(observation_schema.data_type != data_type for observation_schema in observation_schemas[1:]):
        raise ValueError("All configured observations must have the same data type")

    input_size = sum(prod(observation_schema.shape) for observation_schema in observation_schemas)
    return (input_size,), data_type


def _combine_observations(observation: TensorDictBase, observation_keys: tuple[str, ...]) -> torch.Tensor:
    flattened_observations = tuple(observation[key].flatten(start_dim=observation.batch_dims) for key in observation_keys)
    if len(flattened_observations) == 1:
        return flattened_observations[0]
    return torch.cat(flattened_observations, dim=-1)


@dataclass(frozen=True, kw_only=True, slots=True)
class PolicyConfiguration:
    body_factory: BodyFactory
    observation_keys: tuple[str, ...]
    action_key: str
    action_bound_enforcement: ActionBoundEnforcement = ActionBoundEnforcement.BOUND_LOSS
    position_target_mode: PositionTargetMode = PositionTargetMode.ABSOLUTE
    standard_deviation: float = 0.1
    normalization_clip: float | None = 10.0


class Policy(nn.Module):
    def __init__(
        self,
        schema: EnvironmentSchema,
        *,
        configuration: PolicyConfiguration,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        self.observation_keys = configuration.observation_keys
        self.action_key = configuration.action_key
        observation_input_shape, observation_data_type = _get_observation_input_specification(schema, self.observation_keys)
        action_schema = schema.actions[self.action_key]

        self.action_bound_enforcement = configuration.action_bound_enforcement
        self.standard_deviation = configuration.standard_deviation

        lower_bounds, upper_bounds = action_schema.bounds
        lower_bounds = lower_bounds.to(device=device, dtype=action_schema.data_type)
        upper_bounds = upper_bounds.to(device=device, dtype=action_schema.data_type)

        match configuration.position_target_mode:
            case PositionTargetMode.ABSOLUTE:
                normalized_mean_offset = torch.zeros(action_schema.shape, device=device, dtype=action_schema.data_type)
            case PositionTargetMode.DEFAULT_POSE_OFFSET:
                default_value = action_schema.default_value

                default_value = default_value.to(device=device, dtype=action_schema.data_type)
                bound_center = (lower_bounds + upper_bounds) * 0.5
                bound_half_range = (upper_bounds - lower_bounds) * 0.5
                normalized_mean_offset = (default_value - bound_center) / bound_half_range
                if torch.any(normalized_mean_offset <= -1.0) or torch.any(normalized_mean_offset >= 1.0):
                    raise ValueError("The action schema default value must lie within the action bounds")

                if self.action_bound_enforcement is ActionBoundEnforcement.TANH_DISTRIBUTION:
                    normalized_mean_offset = torch.atanh(normalized_mean_offset)

        self.register_buffer("action_lower_bounds", lower_bounds)
        self.register_buffer("action_upper_bounds", upper_bounds)
        self.register_buffer("normalized_mean_offset", normalized_mean_offset)

        self.normalizer = RunningNormalization(
            observation_input_shape,
            clip=configuration.normalization_clip,
            device=device,
            dtype=observation_data_type,
        )
        self.body = configuration.body_factory(
            input_shape=observation_input_shape,
            output_shape=action_schema.shape,
            device=device,
        )

    @property
    def action_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.action_lower_bounds, self.action_upper_bounds

    def forward(self, observation: TensorDictBase) -> torch.Tensor:
        combined_observation = _combine_observations(observation, self.observation_keys)
        normalized_observation = self.normalizer(combined_observation)
        return self.body(normalized_observation)

    def create_distribution(self, mean: torch.Tensor) -> ActionDistribution:
        return ActionDistribution(
            mean + self.normalized_mean_offset,  # We could add the offset in the initial bias, but it's simpler to do it here
            standard_deviation=self.standard_deviation,
            action_bound_enforcement=self.action_bound_enforcement,
            bounds=self.action_bounds,
        )

    def sample_action(self, observation: TensorDictBase, stochastic: bool = True) -> TensorDictBase:
        mean = self(observation)
        distribution = self.create_distribution(mean)
        action = distribution.sample() if stochastic else distribution.action_mean
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
        return distribution.log_prob(action[self.action_key]), distribution.action_mean

    def update_normalizer(self, observation: TensorDictBase) -> None:
        combined_observation = _combine_observations(observation, self.observation_keys)
        self.normalizer.update(combined_observation)


@dataclass(frozen=True, kw_only=True, slots=True)
class ValueFunctionConfiguration:
    body_factory: BodyFactory
    observation_keys: tuple[str, ...]
    normalization_clip: float | None = 10.0


class ValueFunction(nn.Module):
    def __init__(
        self,
        schema: EnvironmentSchema,
        *,
        configuration: ValueFunctionConfiguration,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        self.observation_keys = configuration.observation_keys
        observation_input_shape, observation_data_type = _get_observation_input_specification(schema, self.observation_keys)
        self.normalizer = RunningNormalization(
            observation_input_shape,
            clip=configuration.normalization_clip,
            device=device,
            dtype=observation_data_type,
        )
        self.body = configuration.body_factory(
            input_shape=observation_input_shape,
            output_shape=(1,),
            device=device,
        )

    def forward(self, observation: TensorDictBase) -> torch.Tensor:
        combined_observation = _combine_observations(observation, self.observation_keys)
        normalized_observation = self.normalizer(combined_observation)
        return self.body(normalized_observation).squeeze(-1)

    def update_normalizer(self, observation: TensorDictBase) -> None:
        combined_observation = _combine_observations(observation, self.observation_keys)
        self.normalizer.update(combined_observation)
