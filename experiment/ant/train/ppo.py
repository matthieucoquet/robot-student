# from robot_student.algorithm import PPO
from robot_student.engine.genesis_engine import GenesisEngine
from robot_student.model import MLP

from ..environment import setup_environment


def main():
    engine = GenesisEngine(cuda_backend=True, show_viewer=True)

    environment = setup_environment(engine, environment_count=10)

    model = MLP(environment.schema, "proprioception", "control", device="cuda")

    # algorithm = PPO()
    # algorithm.train()

    observation = environment.reset()
    for _ in range(1000):
        action = model(observation)
        observation = environment.step(action)


if __name__ == "__main__":
    main()
