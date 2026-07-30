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
class RunConfiguration:
    experiment_name: str
    run_name: str
    seed: int
    use_cuda: bool
    iteration_count: int
    checkpoint_interval: int
    metric_log_interval: int = (10,)
    debug_level: int = logging.DEBUG
    environment: EnvironmentFactory
    learner: PPOFactory
    metric_storages: Sequence[MetricStorage] = ((),)
    checkpoint_storages: Sequence[CheckpointStorage] = ((),)


class Run:
    def __init__(self, configuration: RunConfiguration):
        configure_logging(configuration.debug_level)
        set_seed(configuration.seed)

        self._environment = configuration.environment.create(use_cuda=configuration.use_cuda, seed=configuration.seed)
        self._learner = configuration.learner.create(environment=self._environment)

        self._iteration_count = configuration.iteration_count
        self._checkpoint_interval = configuration.checkpoint_interval
        self._metric_log_interval = configuration.metric_log_interval

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

    def train(self):
        self._learner.train()

        for i in range(self._iteration_count):
            metrics = self._learner.update()

            if i % self._metric_log_interval == 0:
                self._log_metrics(metrics, i)

            if i % self._checkpoint_interval == 0 or i == self._iteration_count - 1:
                self._logger.debug(f"Saving checkpoint at interval {i}")
                self._save_checkpoint(
                    self._learner.checkpoint(),
                    i,
                )

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

    def _log_metrics(self, metrics: Mapping[str, ScalarMetric], iteration: int) -> None:
        stored_metrics = {name: value.item() if isinstance(value, torch.Tensor) else value for name, value in metrics.items()}
        for storage in self._metric_storages:
            storage.log(stored_metrics, iteration)

    def _save_checkpoint(self, checkpoint: Checkpoint, iteration: int) -> None:
        for storage in self._checkpoint_storages:
            storage.save(checkpoint, iteration)

    def close(self, exit_code: int = 0) -> None:
        for storage in reversed(self._storages):
            storage.close(exit_code)
