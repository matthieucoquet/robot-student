import logging

import torch
from tensordict import TensorDict, TensorDictBase
from torch import nn

from robot_student.algorithm import PPO
from robot_student.engine.genesis_engine import GenesisEngine
from robot_student.environment.schema import EnvironmentSchema
from robot_student.model import MLP, ActorCritic, create_distribution
from robot_student.model.normalizer import RunningNormalization
from robot_student.util import ExperimentStorage, configure_logging, set_seed

from .environment import setup_environment


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

        return create_distribution(mean)

    def sample_action(self, observation: TensorDictBase) -> TensorDictBase:
        distribution = self(observation)
        action = distribution.sample()
        return TensorDict({self.action_key: action}, batch_size=observation.batch_size, device=action.device)


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


def main():
    with ExperimentStorage("ant", run_name="ppo"):  # as experiment_storage:
        configure_logging(logging.DEBUG)

        seed = 0
        cuda_training = True
        set_seed(seed)
        engine = GenesisEngine(cuda_backend=cuda_training, show_viewer=True, seed=seed)

        environment = setup_environment(engine, environment_count=10)

        device = torch.device("cuda") if cuda_training else torch.device("cpu")

        policy = Policy(environment.schema, device=device)
        value_function = ValueFunction(environment.schema, device=device)
        actor_critic = ActorCritic(policy=policy, value=value_function)

        ppo = PPO(actor_critic)
        ppo.train(iteration_count=10, checkpoint_interval=5)

        observation = environment.reset()
        for _ in range(1000):
            action = policy.sample_action(observation)
            observation = environment.step(action)


if __name__ == "__main__":
    main()
