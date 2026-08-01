import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import torch

from robot_student.algorithm import PPOFactory
from robot_student.util.storage import CheckpointStorage, ExperimentContext, MetricStorage
from robot_student.util.logging import configure_logging
from robot_student.util.seed import set_seed

from .environment_factory import EnvironmentFactory

ScalarMetric = int | float | torch.Tensor
StoredScalarMetric = int | float
Checkpoint = dict[str, Any]


@dataclass(frozen=True, kw_only=True, slots=True)
class EvaluationConfiguration:
    # experiment_name: str
    # run_name: str
    seed: int
    use_cuda: bool
    environment: EnvironmentFactory
    learner: PPOFactory
    debug_level: int = logging.DEBUG
    checkpoint_storages: Sequence[CheckpointStorage] = ()


class Evaluator:
    def __init__(self):
        pass


class Evaluation:
    def __init__(self, configuration: EvaluationConfiguration):
        configure_logging(configuration.debug_level)
        set_seed(configuration.seed)

        self._environment = configuration.environment.create(use_cuda=configuration.use_cuda, seed=configuration.seed)
        self._learner = configuration.learner.create(environment=self._environment)

        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        run_directory = Path.cwd() / "result" / configuration.experiment_name / f"{timestamp}_{configuration.run_name}"
        run_directory.mkdir(parents=True, exist_ok=True)

        context = ExperimentContext(
            experiment_name=configuration.experiment_name,
            run_directory=run_directory,
            seed=configuration.seed,
            device=self._environment.device,
        )

        self._metric_storages = tuple(configuration.metric_storages)
        self._checkpoint_storages = tuple(configuration.checkpoint_storages)
        storages = (*self._metric_storages, *self._checkpoint_storages)
        self._storages = tuple({id(storage): storage for storage in storages}.values())
        for storage in self._storages:
            storage.initialize(context)

        self._logger = logging.getLogger(__name__)

    @torch.inference_mode()
    def __call__(self):
        observation = environment.reset()

        try:
            for _ in range(150):
                action = policy.sample_action(observation, stochastic=True)
                _, _, terminal, truncated, _ = environment.step(action)

                done = torch.logical_or(terminal, truncated)
                observation = environment.reset_done(done)
        finally:
            engine.stop_recording()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exception_type is None:
            exit_code = 0
        elif issubclass(exception_type, KeyboardInterrupt):
            exit_code = 130
        else:
            exit_code = 1
        self.close(exit_code=exit_code)

    def _save_checkpoint(self, checkpoint: Checkpoint, iteration: int) -> None:
        for storage in self._checkpoint_storages:
            storage.save(checkpoint, iteration)

    def close(self, exit_code: int = 0) -> None:
        for storage in reversed(self._storages):
            storage.close(exit_code)
