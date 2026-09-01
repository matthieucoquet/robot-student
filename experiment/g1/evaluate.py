import logging

from robot_student.run import Evaluation, RecordingConfiguration
from robot_student.util import WeightsAndBiasesStorage

from .environment.environment import TrackerEnvironmentFactory
from .learner import get_ppo_factory

if __name__ == "__main__":
    environment = TrackerEnvironmentFactory(headless=False, environment_count=1)
    learner = get_ppo_factory(motion_tracking=True)

    weights_and_biases_storage = WeightsAndBiasesStorage()

    recording_configuration = RecordingConfiguration(position=(2.0, -2.0, 3.0), resolution=(1920, 1080), environment_index=0)

    evaluation = Evaluation(
        experiment_name="g1_deepmimic",
        run_name="ppo",
        run_id="43de9379098fe952",
        seed=0,
        use_cuda=True,
        debug_level=logging.INFO,
        environment_factory=environment,
        learner_factory=learner,
        run_storage=weights_and_biases_storage,
        recording=recording_configuration,
    )
    evaluation.run()
