import torch
from torch.distributions import transforms

from robot_student.model.action import ActionBoundEnforcement


class ActionDistribution(torch.distributions.Independent):
    def __init__(
        self,
        mean: torch.Tensor,
        standard_deviation: float,
        action_offset: torch.Tensor,
        action_scale: torch.Tensor,
        action_bound_enforcement: ActionBoundEnforcement = ActionBoundEnforcement.BOUND_LOSS,
    ) -> None:
        distribution: torch.distributions.Distribution = torch.distributions.Normal(mean, standard_deviation)
        distribution_transforms: list[transforms.Transform] = []
        self.action_mean = mean

        if action_bound_enforcement is ActionBoundEnforcement.TANH_DISTRIBUTION:
            distribution_transforms.append(transforms.TanhTransform(cache_size=1))
            self.action_mean = torch.tanh(self.action_mean)

        distribution_transforms.append(transforms.AffineTransform(loc=action_offset, scale=action_scale))
        self.action_mean = action_offset + action_scale * self.action_mean

        distribution = torch.distributions.TransformedDistribution(distribution, distribution_transforms)

        super().__init__(distribution, reinterpreted_batch_ndims=1)
