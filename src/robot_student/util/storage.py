from collections.abc import Iterator, Mapping
from contextlib import contextmanager
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
class RunContext:
    experiment_name: str
    run_name: str
    run_id: str
    run_directory: Path
    device: torch.device | None
    is_evaluation: bool = False


class Storage(Protocol):
    def initialize(self, context: RunContext) -> None: ...
    def close(self, exit_code: int) -> None: ...


class MetricStorage(Storage, Protocol):
    def log(self, metrics: Mapping[str, StoredScalarMetric], iteration: int) -> None: ...


class CheckpointStorage(Storage, Protocol):
    def save(self, checkpoint: Checkpoint, iteration: int) -> None: ...
    def load(self, iteration: int, *, device: str | torch.device | None = None) -> Checkpoint: ...


class MetricCheckpointStorage(MetricStorage, CheckpointStorage, Protocol):
    pass


class LocalCheckpointStorage(CheckpointStorage):
    def __init__(self) -> None:
        self._checkpoint_directory: Path | None = None

    def initialize(self, context: RunContext) -> None:
        self._checkpoint_directory = context.run_directory / "checkpoints"
        self._checkpoint_directory.mkdir(parents=True, exist_ok=True)
        (context.run_directory / "run_id.txt").write_text(context.run_id, encoding="utf-8")

    def save(self, checkpoint: Checkpoint, iteration: int) -> None:
        checkpoint_path = self._checkpoint_path(iteration)
        torch.save(checkpoint, checkpoint_path)

    def load(
        self,
        iteration: int,
        *,
        device: str | torch.device | None = None,
    ) -> Checkpoint:
        checkpoint_path = self._checkpoint_path(iteration)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint {iteration} not found in {checkpoint_path.parent}")
        return torch.load(checkpoint_path, map_location=device)

    def close(self, exit_code: int) -> None:
        pass

    def _checkpoint_path(self, iteration: int) -> Path:
        if iteration >= 0:
            return self._checkpoint_directory / f"checkpoint_{iteration}.pt"

        latest_iteration = -1
        latest_checkpoint = None

        for checkpoint_path in self._checkpoint_directory.glob("checkpoint_*.pt"):
            iteration_text = checkpoint_path.stem.removeprefix("checkpoint_")
            try:
                iteration = int(iteration_text)
            except ValueError:
                continue

            if iteration > latest_iteration:
                latest_iteration = iteration
                latest_checkpoint = checkpoint_path

        if latest_checkpoint is None:
            raise FileNotFoundError(f"No checkpoints found in {self._checkpoint_directory}")

        return latest_checkpoint


class TensorBoardMetricStorage(MetricStorage, LocalCheckpointStorage):
    def __init__(self) -> None:
        self._writer: SummaryWriter | None = None

    def initialize(self, context: RunContext) -> None:
        if self._writer is not None:
            return
        super().initialize(context)
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


class WeightsAndBiasesStorage(MetricStorage, LocalCheckpointStorage):
    def __init__(self) -> None:
        self._run: wandb.Run | None = None
        self._checkpoint_directory: Path | None = None

    def initialize(self, context: RunContext) -> None:
        if self._run is not None:
            return
        super().initialize(context)

        wandb_directory = context.run_directory / "wandb"
        wandb_directory.mkdir(parents=True, exist_ok=True)

        self.training_id = context.run_id

        self._run = wandb.init(
            project=context.experiment_name,
            name=context.run_name,
            id=None if context.is_evaluation else context.run_id,
            dir=wandb_directory,
            job_type="evaluation" if context.is_evaluation else "training",
            mode="online",
            save_code=False,
        )
        if not context.is_evaluation:
            self._run.define_metric("iteration")
            self._run.define_metric("*", step_metric="iteration")

    def log(self, metrics: Mapping[str, StoredScalarMetric], iteration: int) -> None:
        logged_values = dict(metrics)
        logged_values["iteration"] = iteration
        self._run.log(logged_values)

    def save(self, checkpoint: Checkpoint, iteration: int) -> None:
        super().save(checkpoint, iteration)

        artifact = wandb.Artifact(
            name=f"checkpoint-{self._run.id}",
            type="model",
            metadata={"iteration": iteration},
        )
        artifact.add_file(self._checkpoint_path(iteration), name="checkpoint.pt")
        self._run.log_artifact(
            artifact,
            aliases=["latest", f"iteration-{iteration}"],
        )

    def load(
        self,
        iteration: int,
        *,
        device: str | torch.device | None = None,
    ) -> Checkpoint:
        try:
            return super().load(iteration, device=device)
        except FileNotFoundError:
            artifact_name = f"checkpoint-{self.training_id}"
            artifact_alias = "latest" if iteration == -1 else f"iteration-{iteration}"
            artifact = self._run.use_artifact(f"{artifact_name}:{artifact_alias}")

            checkpoint_directory = Path(artifact.download(root=self._checkpoint_directory))
            checkpoint_path = checkpoint_directory / "checkpoint.pt"
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Artifact {artifact_name}:{artifact_alias} does not contain checkpoint.pt") from None
            return torch.load(checkpoint_path, map_location=device)

    def close(self, exit_code: int) -> None:
        if self._run is not None:
            self._run.finish(exit_code=exit_code)
            self._run = None


@contextmanager
def managed_storage(storage: Storage) -> Iterator[Storage]:
    exit_code = 1

    try:
        yield storage
    except KeyboardInterrupt:
        exit_code = 130
        raise SystemExit(exit_code) from None
    except SystemExit as exception:
        exit_code = exception.code if isinstance(exception.code, int) else int(exception.code is not None)
        raise
    else:
        exit_code = 0
    finally:
        storage.close(exit_code)
