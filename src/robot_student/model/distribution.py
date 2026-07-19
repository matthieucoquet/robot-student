from enum import StrEnum

import torch
from torch.distributions import transforms


class ActionBoundEnforcement(StrEnum):
    TANH_DISTRIBUTION = "tanh_distribution"
    BOUND_LOSS = "bound_loss"


class ActionDistribution(torch.distributions.Independent):
    def __init__(
        self,
        mean: torch.Tensor,
        standard_deviation: float,
        bounds: tuple[torch.Tensor, torch.Tensor],
        action_bound_enforcement: ActionBoundEnforcement = ActionBoundEnforcement.BOUND_LOSS,
    ) -> None:
        distribution: torch.distributions.Distribution = torch.distributions.Normal(mean, standard_deviation)
        distribution_transforms: list[transforms.Transform] = []
        self.action_mean = mean

        if action_bound_enforcement is ActionBoundEnforcement.TANH_DISTRIBUTION:
            distribution_transforms.append(transforms.TanhTransform(cache_size=1))
            self.action_mean = torch.tanh(self.action_mean)

        lower_bounds, upper_bounds = bounds
        center = (lower_bounds + upper_bounds) * 0.5
        half_range = (upper_bounds - lower_bounds) * 0.5
        distribution_transforms.append(transforms.AffineTransform(loc=center, scale=half_range))
        self.action_mean = center + half_range * self.action_mean

        distribution = torch.distributions.TransformedDistribution(distribution, distribution_transforms)

        super().__init__(distribution, reinterpreted_batch_ndims=1)
