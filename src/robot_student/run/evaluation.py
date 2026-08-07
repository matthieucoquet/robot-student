import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from robot_student.algorithm import PPOFactory
from robot_student.util.logging import configure_logging
from robot_student.util.seed import set_seed
from robot_student.util.storage import MetricCheckpointStorage, RunContext, managed_storage

from .environment_factory import EnvironmentFactory

ScalarMetric = int | float | torch.Tensor
StoredScalarMetric = int | float
Checkpoint = dict[str, Any]


@dataclass(kw_only=True)
class RecordingConfiguration:
    position: tuple[float, float, float]
    environment_index: int | None = None
    resolution: tuple[int, int] = (1280, 720)


@dataclass(kw_only=True)
class Evaluation:
    experiment_name: str
    run_name: str
    run_id: str
    seed: int
    use_cuda: bool
    environment_factory: EnvironmentFactory
    learner_factory: PPOFactory
    debug_level: int = logging.DEBUG
    run_storage: MetricCheckpointStorage
    recording: RecordingConfiguration | None = None

    def _setup(self):
        configure_logging(self.debug_level)
        set_seed(self.seed)

        self._engine = self.environment_factory.create_engine(use_cuda=self.use_cuda, seed=self.seed)

        run_directory = self._find_directory()

        if self.recording is not None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            recording_directory = run_directory / "evaluation" / f"recording-{timestamp}.mp4"

            environment_index = self.recording.environment_index
            if environment_index is None:
                environment_index = self.environment_factory.environment_count // 2

            self._engine.setup_recording(
                resolution=self.recording.resolution,
                position=self.recording.position,
                environment_index=environment_index,
                save_to_filename=recording_directory,
            )

        self._environment = self.environment_factory.create_environment(engine=self._engine)
        self._learner = self.learner_factory.create(environment=self._environment)

        context = RunContext(
            experiment_name=self.experiment_name,
            run_name=self.run_name,
            run_id=self.run_id,
            run_directory=run_directory,
            device=self._environment.device,
            is_evaluation=True,
        )
        self.run_storage.initialize(context)

        self._logger = logging.getLogger(__name__)

    @torch.inference_mode()
    def run(self):
        self._setup()

        with managed_storage(self.run_storage):
            checkpoint = self.run_storage.load(iteration=-1, device=self._environment.device)
            policy_state = checkpoint.get("policy")
            policy = self._learner.policy
            policy.load_state_dict(policy_state)
            policy.standard_deviation = 0.01
            policy.eval()

            observation = self._environment.reset()

            try:
                for _ in range(450):
                    action = policy.sample_action(observation, stochastic=True)
                    _, _, terminal, truncated, _ = self._environment.step(action)

                    done = torch.logical_or(terminal, truncated)
                    observation = self._environment.reset_done(done)
            finally:
                self._engine.stop_recording()

    def _find_directory(self):
        result_directory = Path.cwd() / "result" / self.experiment_name
        if result_directory.exists():
            for directory in result_directory.iterdir():
                if directory.is_dir() and directory.name.endswith(self.run_id):
                    return directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        run_directory = result_directory / f"{timestamp}_{self.run_name}_{self.run_id}"
        run_directory.mkdir(parents=True, exist_ok=True)
        return run_directory
