from enum import StrEnum

import torch
from torch.distributions import transforms


class ActionBoundEnforcement(StrEnum):
    TANH_DISTRIBUTION = "tanh_distribution"
    ADDITIONAL_LOSS = "additional_loss"


def create_distribution(
    mean: torch.Tensor,
    standard_deviation: float,
    action_bound_enforcement: ActionBoundEnforcement = ActionBoundEnforcement.ADDITIONAL_LOSS,
    bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.distributions.Distribution:
    normal_distribution = torch.distributions.Normal(mean, standard_deviation)

    action_bound_enforcement = ActionBoundEnforcement(action_bound_enforcement)

    if action_bound_enforcement is ActionBoundEnforcement.TANH_DISTRIBUTION:
        distribution_transforms = [transforms.TanhTransform(cache_size=1)]
        if bounds is not None:
            lower_bounds, upper_bounds = bounds
            lower_bounds = lower_bounds.to(device=mean.device, dtype=mean.dtype)
            upper_bounds = upper_bounds.to(device=mean.device, dtype=mean.dtype)
            half_range = (upper_bounds - lower_bounds) * 0.5
            center = (upper_bounds + lower_bounds) * 0.5
            distribution_transforms.append(transforms.AffineTransform(loc=center, scale=half_range))

        normal_distribution = torch.distributions.TransformedDistribution(normal_distribution, distribution_transforms)

    return torch.distributions.Independent(
        normal_distribution,
        reinterpreted_batch_ndims=1,
    )
