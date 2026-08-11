import logging

from robot_student.run import Evaluation, RecordingConfiguration
from robot_student.util import WeightsAndBiasesStorage

from .environment.environment import G1EnvironmentFactory
from .learner import get_ppo_factory

if __name__ == "__main__":
    environment = G1EnvironmentFactory(headless=False, environment_count=16)
    learner = get_ppo_factory()

    weights_and_biases_storage = WeightsAndBiasesStorage()

    recording_configuration = RecordingConfiguration(position=(2.0, -2.0, 3.0), resolution=(1920, 1080), environment_index=9)

    evaluation = Evaluation(
        experiment_name="g1_walking",
        run_name="ppo",
        run_id="ee3207c09b3f52f5",
        seed=0,
        use_cuda=True,
        debug_level=logging.INFO,
        environment_factory=environment,
        learner_factory=learner,
        run_storage=weights_and_biases_storage,
        recording=recording_configuration,
    )
    evaluation.run()
