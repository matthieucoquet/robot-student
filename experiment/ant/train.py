import logging

from robot_student.run import Training
from robot_student.util import WeightsAndBiasesStorage

from .environment import AntEnvironmentFactory
from .learner import get_ppo_factory

if __name__ == "__main__":
    environment = AntEnvironmentFactory(headless=True, environment_count=2048)
    learner = get_ppo_factory()

    weights_and_biases_storage = WeightsAndBiasesStorage()

    training = Training(
        experiment_name="ant_walking",
        run_name="ppo",
        seed=0,
        use_cuda=True,
        debug_level=logging.INFO,
        iteration_count=10_000,
        checkpoint_interval=1_000,
        metric_log_interval=10,
        environment_factory=environment,
        learner_factory=learner,
        checkpoint_metric_storage=weights_and_biases_storage,
    )
    training.run()
