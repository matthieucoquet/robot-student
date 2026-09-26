import logging

from robot_student.motion import ReferenceSampling
from robot_student.run import Evaluation, RecordingConfiguration
from robot_student.util import WeightsAndBiasesStorage

from .environment.environment import BeyondMimicEnvironmentFactory
from .learner import get_ppo_factory

if __name__ == "__main__":
    environment = BeyondMimicEnvironmentFactory(
        headless=True,
        environment_count=1,
        reference_sampling=ReferenceSampling.ZERO,
        show_reference_motion=True,
        reference_motion_offset=(0.0, 1.0, 0.0),
        enable_randomization=True,
    )
    learner = get_ppo_factory(
        actor_observation_keys=("actor",),
        critic_observation_keys=("critic",),
    )

    weights_and_biases_storage = WeightsAndBiasesStorage()

    recording_configuration = RecordingConfiguration(position=(-1.5, -1.0, 1.5), resolution=(1920, 1080), environment_index=0)

    evaluation = Evaluation(
        experiment_name="g1_beyondmimic",
        run_name="jog_eval",
        run_id="ca42f12a04defe8c",
        seed=0,
        use_cuda=False,
        debug_level=logging.INFO,
        environment_factory=environment,
        learner_factory=learner,
        run_storage=weights_and_biases_storage,
        recording=recording_configuration,
    )
    evaluation.run()
