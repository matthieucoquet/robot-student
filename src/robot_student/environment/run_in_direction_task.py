import torch
from tensordict import TensorDictBase

from robot_student.environment.character_task import CharacterTask


class RunInDirectionTask(CharacterTask):
    def __init__(
        self,
        device: torch.device,
        direction: tuple[float, float] = (1.0, 0.0),
        forward_velocity_weight: float = 1.0,
        control_cost_weight: float = 0.01,
        height_range: tuple[float, float] = (0.2, 1.0),
    ) -> None:

        self._direction = torch.tensor(direction, device=device, dtype=torch.float32)
        self._forward_velocity_weight = forward_velocity_weight
        self._control_cost_weight = control_cost_weight
        self._minimum_healthy_height, self._maximum_healthy_height = height_range

    def step(
        self,
        generalized_positions: torch.Tensor,
        generalized_velocities: torch.Tensor,
        action: TensorDictBase,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        root_height = generalized_positions[..., 2]
        root_height_is_healthy = root_height >= self._minimum_healthy_height
        root_height_is_healthy.logical_and_(root_height <= self._maximum_healthy_height)
        terminal = root_height_is_healthy.logical_not_()

        forward_velocity = torch.sum(generalized_velocities[..., :2] * self._direction, dim=-1)

        control = action["control"]
        control_cost = torch.sum(control.square(), dim=-1)
        stay_alive_reward = root_height_is_healthy * 0.1
        reward = stay_alive_reward + self._forward_velocity_weight * forward_velocity - self._control_cost_weight * control_cost

        return reward, terminal
