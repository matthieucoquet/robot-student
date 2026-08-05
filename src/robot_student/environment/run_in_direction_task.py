import math

import torch

from robot_student.environment.character_task import CharacterTask, CharacterTaskStep
from robot_student.util.geometry import heading_angle


class RunInDirectionTask(CharacterTask):
    def __init__(
        self,
        device: torch.device,
        direction: tuple[float, float] = (1.0, 0.0),
        target_speed: float = 1.0,
        target_speed_weight: float = 1.0,
        facing_direction_weight: float = 0.5,
        control_cost_weight: float = 0.5,
        height_range: tuple[float, float] = (0.2, 1.0),
    ) -> None:
        direction_norm = math.hypot(*direction)
        if not math.isclose(direction_norm, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"direction must be normalized, got norm {direction_norm}")

        self._direction = torch.tensor(direction, device=device, dtype=torch.float32)
        self._direction_heading = math.atan2(direction[1], direction[0])
        self._target_speed = target_speed
        self._target_speed_weight = target_speed_weight
        self._facing_direction_weight = facing_direction_weight
        self._control_cost_weight = control_cost_weight
        self._minimum_healthy_height, self._maximum_healthy_height = height_range

    def step(
        self,
        root_position: torch.Tensor,
        root_rotation: torch.Tensor,
        root_velocity: torch.Tensor,
        normalized_control_forces: torch.Tensor,
    ) -> CharacterTaskStep:
        root_height = root_position[..., 2]
        root_height_is_healthy = root_height >= self._minimum_healthy_height
        root_height_is_healthy.logical_and_(root_height <= self._maximum_healthy_height)
        terminal = ~root_height_is_healthy

        forward_velocity = torch.sum(root_velocity[..., :2] * self._direction, dim=-1)
        target_speed_error = forward_velocity - self._target_speed
        target_speed_reward = target_speed_error.abs().mul_(2.0).neg_().exp_()

        facing_direction_reward = torch.cos(heading_angle(root_rotation) - self._direction_heading)

        control_cost = torch.mean(normalized_control_forces.square(), dim=-1)
        stay_alive_reward = root_height_is_healthy * 0.1
        reward = (
            stay_alive_reward
            + self._target_speed_weight * target_speed_reward
            + self._facing_direction_weight * facing_direction_reward
            - self._control_cost_weight * control_cost
        )

        return CharacterTaskStep(
            reward=reward,
            terminal=terminal,
            transition_metrics={
                "task/forward_velocity_mean": forward_velocity.mean(),
                "task/target_speed_error_mean": target_speed_error.abs().mean(),
                "task/facing_direction_reward_mean": facing_direction_reward.mean(),
                "task/control_cost_mean": control_cost.mean(),
            },
        )
