import logging
from pathlib import Path

from robot_student.motion import ReferenceSampling
from robot_student.run import Evaluation, RecordingConfiguration
from robot_student.util import WeightsAndBiasesStorage

from .environment.environment import BeyondMimicEnvironmentFactory
from .learner import get_beyond_mimic_ppo_factory

if __name__ == "__main__":
    environment = BeyondMimicEnvironmentFactory(
        headless=True,
        environment_count=1,
        motion_path=Path(__file__).parent / "dataset" / "preprocessed" / "BM_Long_Jump_00001.pt",
        reference_sampling=ReferenceSampling.ZERO,
        show_reference_motion=True,
        reference_motion_offset=(0.0, 0.2, 0.0),
        enable_randomization=False,
    )
    learner_configuration = get_beyond_mimic_ppo_factory().configuration

    weights_and_biases_storage = WeightsAndBiasesStorage()

    recording_configuration = RecordingConfiguration(position=(0.45, -0.3, 0.3), resolution=(1920, 1080), environment_index=0)

    evaluation = Evaluation(
        experiment_name="microduck_beyondmimic",
        run_name="eval",
        run_id="ce8bc3a6b6cd4127",
        seed=0,
        use_cuda=False,
        debug_level=logging.INFO,
        environment_factory=environment,
        policy_configuration=learner_configuration.policy,
        run_storage=weights_and_biases_storage,
        recording=recording_configuration,
    )
    evaluation.run()
