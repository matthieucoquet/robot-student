import logging

from robot_student.run import Run, RunConfiguration
from robot_student.util import WeightsAndBiasesStorage

from .environment import AntEnvironmentFactory
from .learner import get_ppo_factory


def run_config():
    environment = AntEnvironmentFactory(headless=True, environment_count=256)
    learner = get_ppo_factory()

    weights_and_biases_storage = WeightsAndBiasesStorage()

    return RunConfiguration(
        experiment_name="ant_walking",
        run_name="ppo",
        seed=0,
        use_cuda=False,
        debug_level=logging.INFO,
        iteration_count=10_000,
        checkpoint_interval=1_000,
        metric_log_interval=10,
        environment=environment,
        learner=learner,
        metric_storages=(weights_and_biases_storage,),
        checkpoint_storages=(weights_and_biases_storage,),
    )


with Run(run_config()) as run:
    run.train()
