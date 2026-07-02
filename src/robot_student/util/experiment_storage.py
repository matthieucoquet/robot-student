from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from types import TracebackType

import torch
from torch.utils.tensorboard import SummaryWriter

ScalarMetric = int | float | torch.Tensor


class CheckpointStorage:
    def __init__(self, checkpoint_directory: Path) -> None:
        self._checkpoint_directory = checkpoint_directory
        self._checkpoint_directory.mkdir(parents=True, exist_ok=True)
        self._checkpoint_name = "checkpoint"

    def save(self, data: dict, iteration: int) -> None:
        checkpoint_path = self._checkpoint_directory / f"{self._checkpoint_name}_{iteration}.pt"
        torch.save(data, checkpoint_path)

    def load(self, iteration: int) -> dict:
        checkpoint_path = self._checkpoint_directory / f"{self._checkpoint_name}_{iteration}.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint {iteration} not found in {self._checkpoint_directory}")
        return torch.load(checkpoint_path)


class TensorboardStorage:
    def __init__(self, log_directory: Path) -> None:
        self._writer = SummaryWriter(log_dir=log_directory)

    def log_scalar(self, name: str, value: ScalarMetric, iteration: int) -> None:
        self._writer.add_scalar(name, value, iteration)

    def log_scalars(self, metrics: Mapping[str, ScalarMetric], iteration: int) -> None:
        for name, value in metrics.items():
            self.log_scalar(name, value, iteration)

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()


class ExperimentStorage:
    def __init__(self, experiment_name: str, run_name: str) -> None:
        self._run_directory = Path.cwd() / "result" / experiment_name / f"{datetime.now().strftime('%Y_%m_%d_%H_%M')}_{run_name}"
        self._run_directory.mkdir(parents=True, exist_ok=True)

        self.checkpoint = CheckpointStorage(self._run_directory / "checkpoints")
        self.metrics = TensorboardStorage(self._run_directory / "tensorboard")

    def __enter__(self) -> "ExperimentStorage":
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self.metrics.close()
