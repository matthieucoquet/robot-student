import random

import torch


def set_seed(seed: int, *, deterministic: bool = False):
    random.seed(seed)
    torch.manual_seed(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
