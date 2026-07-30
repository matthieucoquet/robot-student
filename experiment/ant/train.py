import logging

# from robot_student.util import Experiment, WeightsAndBiasesStorage
from robot_student.run import Run, RunConfiguration

from .environment import AntEnvironmentFactory
from .learner import get_ppo_factory


def run_config():
    environment = AntEnvironmentFactory(headless=False, environment_count=256)
    learner = get_ppo_factory

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
    )


with Run(run_config()) as run:
    run.train()


# After that is old code


# class AntExperiment(Experiment):
#     def __init__(self) -> None:
#         weights_and_biases_storage = WeightsAndBiasesStorage()

#         super().__init__(

#             metric_storages=(weights_and_biases_storage,),
#             checkpoint_storages=(weights_and_biases_storage,),

#         )
