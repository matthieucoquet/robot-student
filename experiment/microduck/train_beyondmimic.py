import logging
from dataclasses import replace

from robot_student.motion import ReferenceSampling
from robot_student.run import EvaluationConfiguration, Training
from robot_student.util import WeightsAndBiasesStorage

from .environment.environment import BeyondMimicEnvironmentFactory
from .learner import get_beyond_mimic_ppo_factory

if __name__ == "__main__":
    environment = BeyondMimicEnvironmentFactory(headless=True, environment_count=2048)
    evaluation = EvaluationConfiguration(
        environment_factory=replace(
            environment,
            environment_count=64,
            headless=True,
            reference_sampling=ReferenceSampling.ZERO,
            show_reference_motion=False,
            enable_randomization=False,
        ),
        seed=0,
        maximum_steps=1_000,
    )

    training = Training(
        experiment_name="microduck_beyondmimic",
        run_name="beyondmimic_walk",
        seed=0,
        use_cuda=True,
        debug_level=logging.INFO,
        iteration_count=12_000,
        checkpoint_interval=500,
        metric_log_interval=50,
        environment_factory=environment,
        evaluation=evaluation,
        learner_factory=get_beyond_mimic_ppo_factory(compile_models=True),
        run_storage=WeightsAndBiasesStorage(),
        profiling=None,
    )
    training.run()
