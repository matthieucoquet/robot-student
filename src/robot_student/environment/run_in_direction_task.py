import math

import torch
from tensordict import TensorDictBase

from robot_student.environment.character_task import CharacterTask, CharacterTaskStep


class RunInDirectionTask(CharacterTask):
    def __init__(
        self,
        device: torch.device,
        direction: tuple[float, float] = (1.0, 0.0),
        forward_velocity_weight: float = 1.0,
        control_cost_weight: float = 0.01,
        height_range: tuple[float, float] = (0.2, 1.0),
    ) -> None:
        direction_norm = math.hypot(*direction)
        if not math.isclose(direction_norm, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"direction must be normalized, got norm {direction_norm}")

        self._direction = torch.tensor(direction, device=device, dtype=torch.float32)
        self._forward_velocity_weight = forward_velocity_weight
        self._control_cost_weight = control_cost_weight
        self._minimum_healthy_height, self._maximum_healthy_height = height_range

    def step(
        self,
        generalized_positions: torch.Tensor,
        generalized_velocities: torch.Tensor,
        action: TensorDictBase,
    ) -> CharacterTaskStep:
        root_height = generalized_positions[..., 2]
        root_height_is_healthy = root_height >= self._minimum_healthy_height
        root_height_is_healthy.logical_and_(root_height <= self._maximum_healthy_height)
        terminal = ~root_height_is_healthy

        forward_velocity = torch.sum(generalized_velocities[..., :2] * self._direction, dim=-1)

        control = action["control"]
        control_cost = torch.sum(control.square(), dim=-1)
        stay_alive_reward = root_height_is_healthy * 0.5
        reward = stay_alive_reward + self._forward_velocity_weight * forward_velocity - self._control_cost_weight * control_cost

        return CharacterTaskStep(
            reward=reward,
            terminal=terminal,
            transition_metrics={"task/forward_velocity_mean": forward_velocity.mean()},
        )
