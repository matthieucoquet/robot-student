import logging

import torch

from robot_student.algorithm import PPO
from robot_student.algorithm.rollout_buffer import RolloutBuffer
from robot_student.engine.genesis_engine import GenesisEngine
from robot_student.util import Experiment

from .environment import setup_environment
from .model import Policy, ValueFunction


class AntExperiment(Experiment):
    def __init__(self):
        super().__init__(
            experiment_name="ant",
            run_name="ppo",
            seed=0,
            debug_level=logging.DEBUG,
            device=torch.device("cuda"),
        )

    def setup(self):
        engine = GenesisEngine(cuda_backend=self.is_on_cuda, show_viewer=True, seed=self.seed)

        environment_count = 10
        environment = setup_environment(engine, environment_count=environment_count)

        policy = Policy(environment.schema, device=self.device)
        value_function = ValueFunction(environment.schema, device=self.device)

        learning_rate = 3e-4
        policy_optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
        value_optimizer = torch.optim.Adam(value_function.parameters(), lr=learning_rate)

        rollout_buffer = RolloutBuffer(
            schema=environment.schema, rollout_length=32, environment_count=environment_count, device=self.device
        )

        self.algo = PPO(
            environment=environment,
            policy=policy,
            value_function=value_function,
            policy_optimizer=policy_optimizer,
            value_optimizer=value_optimizer,
            rollout_buffer=rollout_buffer,
        )

    def launch(self):
        self.algo.train(self, iteration_count=10, checkpoint_interval=5)


if __name__ == "__main__":
    with AntExperiment() as experiment:
        experiment.setup()
        experiment.launch()
