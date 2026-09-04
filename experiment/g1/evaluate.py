import logging

from robot_student.run import Evaluation, RecordingConfiguration
from robot_student.util import WeightsAndBiasesStorage

from .environment.environment import TrackerEnvironmentFactory
from .learner import get_ppo_factory

if __name__ == "__main__":
    environment = TrackerEnvironmentFactory(
        headless=True,
        environment_count=1,
        random_reference_sampling=False,
        show_reference_motion=True,
    )
    learner = get_ppo_factory(motion_tracking=True)

    weights_and_biases_storage = WeightsAndBiasesStorage()

    recording_configuration = RecordingConfiguration(position=(-1.5, -1.0, 1.5), resolution=(1920, 1080), environment_index=0)

    evaluation = Evaluation(
        experiment_name="g1_deepmimic",
        run_name="ppo",
        run_id="ac75ca8564f7f3b8",
        seed=0,
        use_cuda=False,
        debug_level=logging.INFO,
        environment_factory=environment,
        learner_factory=learner,
        run_storage=weights_and_biases_storage,
        recording=recording_configuration,
    )
    evaluation.run()
