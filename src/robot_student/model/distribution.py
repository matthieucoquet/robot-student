import torch


def create_distribution(
    mean: torch.Tensor,
    standard_deviation: float = 1.0,
) -> torch.distributions.Distribution:
    return torch.distributions.Independent(
        torch.distributions.Normal(mean, standard_deviation),
        reinterpreted_batch_ndims=1,
    )
