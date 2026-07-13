import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, Self

import torch
import wandb
from torch.utils.tensorboard import SummaryWriter

from .logging import configure_logging
from .seed import set_seed

ScalarMetric = int | float | torch.Tensor
StoredScalarMetric = int | float
Checkpoint = dict[str, Any]


@dataclass(frozen=True)
class ExperimentContext:
    experiment_name: str
    run_directory: Path
    seed: int
    device: torch.device | None


class MetricStorage(Protocol):
    def initialize(self, context: ExperimentContext) -> None: ...
    def log(self, metrics: Mapping[str, StoredScalarMetric], iteration: int) -> None: ...
    def close(self, exit_code: int) -> None: ...


class CheckpointStorage(Protocol):
    def initialize(self, context: ExperimentContext) -> None: ...
    def save(self, checkpoint: Checkpoint, iteration: int) -> None: ...
    def close(self, exit_code: int) -> None: ...


class LocalCheckpointStorage:
    def __init__(self) -> None:
        self._checkpoint_directory: Path | None = None

    def initialize(self, context: ExperimentContext) -> None:
        self._checkpoint_directory = context.run_directory / "checkpoints"
        self._checkpoint_directory.mkdir(parents=True, exist_ok=True)

    def save(self, checkpoint: Checkpoint, iteration: int) -> None:
        checkpoint_path = self._checkpoint_path(iteration)
        torch.save(checkpoint, checkpoint_path)

    def load(
        self,
        iteration: int,
        *,
        map_location: str | torch.device | None = None,
    ) -> Checkpoint:
        checkpoint_path = self._checkpoint_path(iteration)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint {iteration} not found in {checkpoint_path.parent}")
        return torch.load(checkpoint_path, map_location=map_location)

    def close(self, exit_code: int) -> None:
        pass

    def _checkpoint_path(self, iteration: int) -> Path:
        return self._checkpoint_directory / f"checkpoint_{iteration}.pt"


class TensorBoardMetricStorage:
    def __init__(self) -> None:
        self._writer: SummaryWriter | None = None

    def initialize(self, context: ExperimentContext) -> None:
        if self._writer is not None:
            return
        log_directory = context.run_directory / "tensorboard"
        self._writer = SummaryWriter(log_dir=log_directory)

    def log(self, metrics: Mapping[str, StoredScalarMetric], iteration: int) -> None:
        for name, value in metrics.items():
            self._writer.add_scalar(name, value, iteration)

    def close(self, exit_code: int) -> None:
        if self._writer is None:
            return
        self._writer.flush()
        self._writer.close()
        self._writer = None


class WeightsAndBiasesStorage:
    def __init__(
        self,
        project: str = "robot-student-ppo",
        configuration: Mapping[str, object] | None = None,
    ) -> None:
        self._project = project
        self._configuration = dict(configuration or {})
        self._run: wandb.Run | None = None
        self._checkpoint_directory: Path | None = None

    def initialize(self, context: ExperimentContext) -> None:
        if self._run is not None:
            return

        wandb_directory = context.run_directory / "wandb"
        wandb_directory.mkdir(parents=True, exist_ok=True)
        configuration = dict(self._configuration)
        configuration.update(
            {
                "experiment_name": context.experiment_name,
                "seed": context.seed,
                "device": str(context.device),
            }
        )

        self._run = wandb.init(
            project=self._project,
            name=context.run_directory.name,
            dir=wandb_directory,
            config=configuration,
            mode="online",
            force=True,
            save_code=False,
        )
        self._run.define_metric("iteration")
        self._run.define_metric("*", step_metric="iteration")
        self._checkpoint_directory = wandb_directory / "checkpoints"
        self._checkpoint_directory.mkdir(parents=True, exist_ok=True)

    def log(self, metrics: Mapping[str, StoredScalarMetric], iteration: int) -> None:
        logged_values = dict(metrics)
        logged_values["iteration"] = iteration
        self._run.log(logged_values)

    def save(self, checkpoint: Checkpoint, iteration: int) -> None:
        checkpoint_path = self._checkpoint_directory / f"checkpoint_{iteration}.pt"
        torch.save(checkpoint, checkpoint_path)

        artifact = wandb.Artifact(
            name=f"checkpoint-{self._run.id}",
            type="model",
            metadata={"iteration": iteration},
        )
        artifact.add_file(checkpoint_path, name="checkpoint.pt")
        self._run.log_artifact(
            artifact,
            aliases=["latest", f"iteration-{iteration}"],
        )

    def close(self, exit_code: int) -> None:
        if self._run is not None:
            self._run.finish(exit_code=exit_code)
            self._run = None


class Experiment:
    def __init__(
        self,
        experiment_name: str,
        run_name: str,
        metric_storages: Sequence[MetricStorage] = (),
        checkpoint_storages: Sequence[CheckpointStorage] = (),
        seed: int = 0,
        debug_level: int = logging.DEBUG,
        device: torch.device | None = None,
    ) -> None:
        configure_logging(debug_level)
        set_seed(seed)
        self.seed = seed
        self.device = device

        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        run_directory = Path.cwd() / "result" / experiment_name / f"{timestamp}_{run_name}"
        run_directory.mkdir(parents=True, exist_ok=True)

        context = ExperimentContext(
            experiment_name=experiment_name,
            run_directory=run_directory,
            seed=seed,
            device=device,
        )

        self._metric_storages = tuple(metric_storages)
        self._checkpoint_storages = tuple(checkpoint_storages)
        storages = (*self._metric_storages, *self._checkpoint_storages)
        self._storages = tuple({id(storage): storage for storage in storages}.values())
        for storage in self._storages:
            storage.initialize(context)

    @property
    def is_on_cuda(self) -> bool:
        return self.device is not None and self.device.type == "cuda"

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close(exit_code=exception_type is not None)

    def log_metrics(self, metrics: Mapping[str, ScalarMetric], iteration: int) -> None:
        stored_metrics = {name: value.item() if isinstance(value, torch.Tensor) else value for name, value in metrics.items()}
        for storage in self._metric_storages:
            storage.log(stored_metrics, iteration)

    def save_checkpoint(self, checkpoint: Checkpoint, iteration: int) -> None:
        for storage in self._checkpoint_storages:
            storage.save(checkpoint, iteration)

    def close(self, exit_code: int = 0) -> None:
        for storage in reversed(self._storages):
            storage.close(exit_code)
