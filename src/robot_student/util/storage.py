from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
import wandb
from torch.utils.tensorboard import SummaryWriter

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
            name=context.run_directory.name,  # TODO maybe just the name
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
