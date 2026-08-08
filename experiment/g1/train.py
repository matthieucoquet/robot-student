import logging

from robot_student.run import Training, ProfilingConfiguration
from robot_student.util import WeightsAndBiasesStorage

from .environment import G1EnvironmentFactory
from .learner import get_ppo_factory

if __name__ == "__main__":
    environment = G1EnvironmentFactory(headless=True, environment_count=4096)
    learner = get_ppo_factory(compile_models=True)

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
        experiment_name="g1_walking",
        run_name="ppo",
        seed=0,
        use_cuda=True,
        debug_level=logging.INFO,
        iteration_count=10_000,
        checkpoint_interval=100,
        metric_log_interval=10,
        environment_factory=environment,
        learner_factory=learner,
        run_storage=weights_and_biases_storage,
        profiling=None,
    )
    training.run()
