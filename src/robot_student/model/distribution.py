import torch
from torch.distributions import transforms

from robot_student.model.action import ActionBoundEnforcement


def transform_action_mean(
    mean: torch.Tensor,
    action_offset: torch.Tensor,
    action_scale: torch.Tensor,
    action_bound_enforcement: ActionBoundEnforcement,
) -> torch.Tensor:
    if action_bound_enforcement is ActionBoundEnforcement.TANH_DISTRIBUTION:
        mean = torch.tanh(mean)
    return action_offset + action_scale * mean


class ActionDistribution(torch.distributions.Independent):
    def __init__(
        self,
        mean: torch.Tensor,
        standard_deviation: torch.Tensor,
        action_offset: torch.Tensor,
        action_scale: torch.Tensor,
        action_bound_enforcement: ActionBoundEnforcement = ActionBoundEnforcement.BOUND_LOSS,
    ) -> None:
        distribution: torch.distributions.Distribution = torch.distributions.Normal(mean, standard_deviation)
        self._normal_distribution = distribution
        self._action_scale = action_scale
        self._tanh_transform = None
        distribution_transforms: list[transforms.Transform] = []
        self.action_mean = transform_action_mean(mean, action_offset, action_scale, action_bound_enforcement)

        if action_bound_enforcement is ActionBoundEnforcement.TANH_DISTRIBUTION:
            self._tanh_transform = transforms.TanhTransform(cache_size=1)
            distribution_transforms.append(self._tanh_transform)

        distribution_transforms.append(transforms.AffineTransform(loc=action_offset, scale=action_scale))

        distribution = torch.distributions.TransformedDistribution(distribution, distribution_transforms)

        super().__init__(distribution, reinterpreted_batch_ndims=1)

    def entropy(self) -> torch.Tensor:
        """Return action entropy, estimating the tanh correction with a reparameterized sample."""
        entropy = self._normal_distribution.entropy() + self._action_scale.abs().log()
        if self._tanh_transform is not None:
            sample = self._normal_distribution.rsample()
            entropy = entropy + self._tanh_transform.log_abs_det_jacobian(sample, self._tanh_transform(sample))
        return entropy.sum(dim=-1)
