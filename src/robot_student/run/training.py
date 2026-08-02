import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from secrets import token_hex
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
class Training:
    experiment_name: str
    run_name: str
    seed: int
    use_cuda: bool
    iteration_count: int
    checkpoint_interval: int
    environment_factory: EnvironmentFactory
    learner_factory: PPOFactory
    run_id: str | None = None
    metric_log_interval: int = 10
    debug_level: int = logging.DEBUG
    run_storage: MetricCheckpointStorage

    def _setup(self):
        configure_logging(self.debug_level)
        set_seed(self.seed)

        self._engine = self.environment_factory.create_engine(use_cuda=self.use_cuda, seed=self.seed)
        self._environment = self.environment_factory.create_environment(engine=self._engine)
        self._learner = self.learner_factory.create(environment=self._environment)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        if self.run_id is None:
            self.run_id = token_hex(8)
        run_directory = Path.cwd() / "result" / self.experiment_name / f"{timestamp}_{self.run_name}_{self.run_id}"
        run_directory.mkdir(parents=True, exist_ok=True)

        context = RunContext(
            experiment_name=self.experiment_name,
            run_name=self.run_name,
            run_id=self.run_id,
            run_directory=run_directory,
            device=self._environment.device,
        )
        self.run_storage.initialize(context)

        self._logger = logging.getLogger(__name__)

    def run(self):
        self._setup()

        with managed_storage(self.run_storage):
            self._learner.train()

            for i in range(self.iteration_count):
                metrics = self._learner.update()

                if i % self.metric_log_interval == 0:
                    self._log_metrics(metrics, i)

                if i % self.checkpoint_interval == 0 or i == self.iteration_count - 1:
                    self._logger.debug(f"Saving checkpoint at interval {i}")
                    self.run_storage.save(self._learner.checkpoint(), i)

    def _log_metrics(self, metrics: Mapping[str, ScalarMetric], iteration: int) -> None:
        stored_metrics = {name: value.item() if isinstance(value, torch.Tensor) else value for name, value in metrics.items()}
        self.run_storage.log(stored_metrics, iteration)

    def close(self, exit_code: int = 0) -> None:
        self.run_storage.close(exit_code)
