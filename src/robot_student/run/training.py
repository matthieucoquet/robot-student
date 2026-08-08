import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
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


@dataclass(frozen=True, kw_only=True, slots=True)
class ProfilingConfiguration:
    skip_first_iterations: int = 5
    warmup_iterations: int = 1
    active_iterations: int = 3
    record_shapes: bool = True
    profile_memory: bool = True
    with_stack: bool = True


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
    profiling: ProfilingConfiguration | None = None

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
        self._run_directory = run_directory

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

            with self._managed_profiler() as profiler:
                for i in range(self.iteration_count):
                    metrics = self._learner.update()

                    if i % self.metric_log_interval == 0:
                        self._log_metrics(metrics, i)

                    if i % self.checkpoint_interval == 0 or i == self.iteration_count - 1:
                        self._logger.debug(f"Saving checkpoint at interval {i}")
                        self.run_storage.save(self._learner.checkpoint(), i)

                    if profiler is not None:
                        profiler.step()

    @contextmanager
    def _managed_profiler(self) -> Iterator[torch.profiler.profile | None]:
        configuration = self.profiling
        if configuration is None:
            yield None
            return

        supported_activities = torch.profiler.supported_activities()
        activities = [torch.profiler.ProfilerActivity.CPU]
        if self._environment.device.type == "cuda" and torch.profiler.ProfilerActivity.CUDA in supported_activities:
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        tensorboard_directory = self._run_directory / "tensorboard"
        trace_directory = tensorboard_directory / "profiler"
        trace_directory.mkdir(parents=True, exist_ok=True)
        self._logger.info("PyTorch profiler traces will be written to %s", trace_directory)
        self._logger.info("Inspect them with: uv run tensorboard --logdir %s", tensorboard_directory)

        schedule = torch.profiler.schedule(
            wait=0,
            warmup=configuration.warmup_iterations,
            active=configuration.active_iterations,
            repeat=1,
            skip_first=configuration.skip_first_iterations,
        )
        trace_handler = torch.profiler.tensorboard_trace_handler(str(trace_directory))
        with torch.profiler.profile(
            activities=activities,
            schedule=schedule,
            on_trace_ready=trace_handler,
            record_shapes=configuration.record_shapes,
            profile_memory=configuration.profile_memory,
            with_stack=configuration.with_stack,
        ) as profiler:
            yield profiler

    def _log_metrics(self, metrics: Mapping[str, ScalarMetric], iteration: int) -> None:
        stored_metrics = {name: value.item() if isinstance(value, torch.Tensor) else value for name, value in metrics.items()}
        self.run_storage.log(stored_metrics, iteration)

    def close(self, exit_code: int = 0) -> None:
        self.run_storage.close(exit_code)
