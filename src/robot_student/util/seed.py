import random
from collections.abc import Generator
from contextlib import contextmanager

import numpy as np
import torch


def set_seed(seed: int, *, deterministic: bool = False):
    random.seed(seed)
    torch.manual_seed(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)


@contextmanager
def isolated_random_seed(seed: int) -> Generator[None]:
    """Seed evaluation without consuming the training process's random streams."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            random.seed(seed)
            np.random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            if cuda_devices:
                torch.cuda.manual_seed_all(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
