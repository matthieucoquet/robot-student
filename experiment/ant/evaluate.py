import logging

from robot_student.run import Evaluation, RecordingConfiguration
from robot_student.util import WeightsAndBiasesStorage

from .environment import AntEnvironmentFactory
from .learner import get_ppo_factory

if __name__ == "__main__":
    environment = AntEnvironmentFactory(headless=False, environment_count=4)
    learner = get_ppo_factory()

    weights_and_biases_storage = WeightsAndBiasesStorage()

    recording_configuration = RecordingConfiguration(
        position=(8.0, -8.0, 12.0),
        resolution=(1280, 720),
    )

    evaluation = Evaluation(
        experiment_name="ant_walking",
        run_name="ppo",
        run_id="0588ffade70411c9",
        seed=0,
        use_cuda=True,
        debug_level=logging.INFO,
        environment_factory=environment,
        learner_factory=learner,
        run_storage=weights_and_biases_storage,
        recording=recording_configuration,
    )
    evaluation.run()
