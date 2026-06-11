from pathlib import Path

# from robot_student.algorithm import PPO
from robot_student.environment import CharacterEnvironment
from robot_student.models import MLP


def main():
    # experiment_directory = Path("./experiments/")

    device = "cuda"

    environment = CharacterEnvironment(Path("./data/mjcf/inverted_pendulum.xml"), environment_count=10, device=device)

    # algorithm = PPO()
    # algorithm.train()

    model = MLP(environment.schema, "proprioception", "control", device=device)

    observation = environment.reset()
    for _ in range(1000):
        action = model(observation)
        observation = environment.step(action)


if __name__ == "__main__":
    main()
