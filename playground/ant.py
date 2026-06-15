from pathlib import Path

from robot_student.engine.genesis_engine import GenesisEngine

# from robot_student.algorithm import PPO
from robot_student.environment import CharacterEnvironment
from robot_student.model import MLP


class Experiment:
    def __init__(self):
        # experiment_directory = Path("./experiments/")
        self._setup_engine()
        self._setup_environment()
        self._setup_model()

    def _setup_engine(self):
        self._engine = GenesisEngine(cuda_backend=True, show_viewer=True)

    def _setup_environment(self):
        self._environment = CharacterEnvironment(self._engine, Path("./data/mjcf/ant.xml"), environment_count=10)

    def _setup_model(self):
        self._model = MLP(self._environment.schema, "proprioception", "control", device="cuda")

    def train(self):

        # algorithm = PPO()
        # algorithm.train()

        observation = self._environment.reset()
        for _ in range(1000):
            action = self._model(observation)
            observation = self._environment.step(action)


if __name__ == "__main__":
    experiment = Experiment()
    experiment.train()
