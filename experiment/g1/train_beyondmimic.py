import logging
from dataclasses import replace

from robot_student.motion import ReferenceSampling
from robot_student.run import EvaluationConfiguration, Training
from robot_student.util import WeightsAndBiasesStorage

from .environment.environment import BeyondMimicEnvironmentFactory
from .learner import get_ppo_factory

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

    learner = get_ppo_factory(
        actor_observation_keys=["actor"],
        critic_observation_keys=["critic"],
        compile_models=True,
    )

    weights_and_biases_storage = WeightsAndBiasesStorage()

    # profiling = ProfilingConfiguration(
    #     skip_first_iterations=5,
    #     warmup_iterations=2,
    #     active_iterations=3,
    #     record_shapes=False,
    #     profile_memory=False,
    #     with_stack=True,
    # )

    training = Training(
        experiment_name="g1_beyondmimic",
        run_name="beyondmimic_walk",
        seed=0,
        use_cuda=True,
        debug_level=logging.INFO,
        iteration_count=12_000,
        checkpoint_interval=500,
        metric_log_interval=50,
        environment_factory=environment,
        evaluation=evaluation,
        learner_factory=learner,
        run_storage=weights_and_biases_storage,
        profiling=None,
    )
    training.run()
