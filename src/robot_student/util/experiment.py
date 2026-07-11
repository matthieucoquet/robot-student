import logging
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Self

import torch
from torch.utils.tensorboard import SummaryWriter

from .logging import configure_logging
from .seed import set_seed

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
        if isinstance(value, torch.Tensor):
            value = value.item()
        self._writer.add_scalar(name, value, iteration)

    def log_scalars(self, metrics: Mapping[str, ScalarMetric], iteration: int) -> None:
        for name, value in metrics.items():
            self.log_scalar(name, value, iteration)

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self.flush()
        self._writer.close()


class Experiment:
    def __init__(
        self,
        experiment_name: str,
        run_name: str,
        seed: int = 0,
        debug_level: int = logging.DEBUG,
        device: torch.device | None = None,
    ) -> None:
        configure_logging(debug_level)
        set_seed(seed)
        self.seed = seed
        self.device = device

        self._run_directory = Path.cwd() / "result" / experiment_name / f"{datetime.now().strftime('%Y_%m_%d_%H_%M')}_{run_name}"
        self._run_directory.mkdir(parents=True, exist_ok=True)

        self.checkpoint = CheckpointStorage(self._run_directory / "checkpoints")
        self.metrics = TensorboardStorage(self._run_directory / "tensorboard")

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
        self.close()

    def close(self) -> None:
        self.metrics.close()
