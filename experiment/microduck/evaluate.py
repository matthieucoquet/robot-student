import logging

from robot_student.run import Evaluation, RecordingConfiguration
from robot_student.util import WeightsAndBiasesStorage

from .environment.environment import PPOEnvironmentFactory
from .learner import get_ppo_factory

if __name__ == "__main__":
    environment = PPOEnvironmentFactory(
        headless=False,
        environment_count=1,
    )
    learner = get_ppo_factory(
        actor_observation_keys=("proprioception",),
        critic_observation_keys=("proprioception",),
    )

    weights_and_biases_storage = WeightsAndBiasesStorage()

    recording_configuration = RecordingConfiguration(position=(-1.0, 1.5, 1.5), resolution=(1920, 1080), environment_index=0)

    evaluation = Evaluation(
        experiment_name="md_walking",
        run_name="eval",
        run_id="239eb724afbd0c85",
        seed=0,
        use_cuda=False,
        debug_level=logging.INFO,
        environment_factory=environment,
        learner_factory=learner,
        run_storage=weights_and_biases_storage,
        recording=recording_configuration,
    )
    evaluation.run()
