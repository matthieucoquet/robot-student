from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch
import wandb
from genesis.vis.camera import Camera
from tensordict import TensorDict, TensorDictBase

from robot_student.engine.genesis_engine import GenesisEngine
from robot_student.environment import Environment
from robot_student.model import ActionBoundEnforcement

from .environment import setup_environment
from .model import Policy

WEIGHTS_AND_BIASES_ENTITY = "mcoquet"
WEIGHTS_AND_BIASES_PROJECT = "robot-student-ppo"
WEIGHTS_AND_BIASES_RUN_ID = "5p9wbvd9"
WEIGHTS_AND_BIASES_ARTIFACT_ALIAS = "latest"


def _download_policy_state() -> Mapping[str, Any]:
    artifact_reference = (
        f"{WEIGHTS_AND_BIASES_ENTITY}/{WEIGHTS_AND_BIASES_PROJECT}/"
        f"checkpoint-{WEIGHTS_AND_BIASES_RUN_ID}:{WEIGHTS_AND_BIASES_ARTIFACT_ALIAS}"
    )
    artifact = wandb.Api().artifact(artifact_reference, type="model")

    with TemporaryDirectory(prefix="robot-student-ant-checkpoint-") as temporary_directory:
        artifact_directory = Path(artifact.download(root=temporary_directory))
        checkpoint_path = artifact_directory / "checkpoint.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Artifact {artifact_reference} does not contain checkpoint.pt")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    policy_state = checkpoint.get("policy")
    if not isinstance(policy_state, Mapping):
        raise TypeError(f"Checkpoint in artifact {artifact_reference} does not contain a policy state")
    return policy_state


def _select_action(policy: Policy, observation: TensorDictBase) -> TensorDictBase:
    action_mean = policy.create_distribution(policy(observation)).action_mean
    return TensorDict(
        {policy.action_key: action_mean},
        batch_size=observation.batch_size,
        device=action_mean.device,
    )


@torch.inference_mode()
def _visualize(
    policy: Policy,
    environment: Environment,
    engine: Camera,
) -> None:
    observation = environment.reset()

    try:
        for _ in range(150):
            action = _select_action(policy, observation)
            _, _, terminal, truncated, _ = environment.step(action)

            done = torch.logical_or(terminal, truncated)
            observation = environment.reset_done(done)
    finally:
        engine.stop_recording()


def main() -> None:
    policy_state = _download_policy_state()

    engine = GenesisEngine(
        cuda_backend=False,
        show_viewer=False,
        seed=0,
        control_frequency=30,
        simulation_frequency=120,
    )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    recording_directory = Path("result/ant") / f"evaluation-{WEIGHTS_AND_BIASES_RUN_ID}-{timestamp}.mp4"
    engine.setup_recording(
        resolution=(1280, 720),
        position=(8.0, -8.0, 12.0),
        environment_index=27,
        save_to_filename=recording_directory,
    )

    environment = setup_environment(engine, device=engine.device, environment_count=64)
    policy = Policy(
        environment.schema,
        device=engine.device,
        action_bound_enforcement=ActionBoundEnforcement.BOUND_LOSS,
    )
    policy.load_state_dict(policy_state)
    policy.eval()

    _visualize(
        policy,
        environment,
        engine,
    )


if __name__ == "__main__":
    main()
