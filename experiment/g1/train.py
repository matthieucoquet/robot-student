import logging

import torch

from robot_student.algorithm import PPO
from robot_student.algorithm.rollout_buffer import RolloutBuffer
from robot_student.engine.genesis_engine import GenesisEngine
from robot_student.model import ActionBoundEnforcement
from robot_student.util import Experiment, WeightsAndBiasesStorage

from .environment import setup_environment
from .model import Policy, ValueFunction


class G1Experiment(Experiment):
    def __init__(self) -> None:
        weights_and_biases_storage = WeightsAndBiasesStorage()
        seed = 0
        engine = GenesisEngine(
            cuda_backend=False,
            show_viewer=False,
            seed=seed,
            control_frequency=30,
            simulation_frequency=120,
        )
        super().__init__(
            experiment_name="g1",
            run_name="ppo",
            engine=engine,
            metric_storages=(weights_and_biases_storage,),
            checkpoint_storages=(weights_and_biases_storage,),
            seed=seed,
            debug_level=logging.INFO,
        )

    def setup(self) -> None:
        environment_count = 256
        environment = setup_environment(self.engine, device=self.device, environment_count=environment_count)

        action_bound_enforcement = ActionBoundEnforcement.BOUND_LOSS

        policy = Policy(
            environment.schema,
            device=self.device,
            action_bound_enforcement=action_bound_enforcement,
        )
        value_function = ValueFunction(environment.schema, device=self.device)

        learning_rate = 1e-4
        policy_optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
        value_optimizer = torch.optim.Adam(value_function.parameters(), lr=learning_rate)

        rollout_buffer = RolloutBuffer(
            schema=environment.schema, rollout_length=32, environment_count=environment_count, device=self.device
        )

        self.algorithm = PPO(
            environment=environment,
            policy=policy,
            value_function=value_function,
            policy_optimizer=policy_optimizer,
            value_optimizer=value_optimizer,
            rollout_buffer=rollout_buffer,
        )

    def launch(self):
        self.algorithm.train(self, iteration_count=10000, checkpoint_interval=100)


if __name__ == "__main__":
    with G1Experiment() as experiment:
        experiment.setup()
        experiment.launch()
