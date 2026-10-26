import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import genesis as gs
import torch
from tensordict import TensorDict, TensorDictBase

from robot_student.engine.control_mode import ControlMode
from robot_student.engine.robot import CommandDelayConfiguration, DomainRandomizationConfiguration
from robot_student.environment.environment import Environment
from robot_student.environment.schema import EnvironmentSchema, TensorSchema
from robot_student.environment.task.task import Task

if TYPE_CHECKING:
    from robot_student.engine.genesis_engine import GenesisEngine


@dataclass(frozen=True, kw_only=True, slots=True)
class PushConfiguration:
    interval_seconds: float = 2.0
    linear_velocity_half_width: tuple[float, float, float] = (0.5, 0.5, 0.2)  # Meters per second.
    angular_velocity_half_width: tuple[float, float, float] = (0.52, 0.52, 0.78)  # Radians per second.


class RobotEnvironment(Environment):
    def __init__(
        self,
        engine: "GenesisEngine",
        xml_path: Path,
        control_mode: ControlMode,
        task: Task,
        control_frequency: int,
        initial_pose: Sequence[float],
        maximum_episode_steps: int = 1_000,
        *,
        ground_friction: float | None = None,
        domain_randomization_configuration: DomainRandomizationConfiguration | None = None,
        command_delay_configuration: CommandDelayConfiguration | None = None,
        push_configuration: PushConfiguration | None = None,
    ) -> None:
        self._engine = engine
        self._task = task
        self._simulation_steps_per_control_step = engine.simulation_frequency // control_frequency
        self._engine.add_ground_plane(friction=ground_friction)
        self._robot = engine.add_robot(
            xml_path,
            control_mode=control_mode,
            domain_randomization_configuration=domain_randomization_configuration,
            command_delay_configuration=command_delay_configuration,
        )

        device = engine.device
        self._task.setup_scene(engine)
        self._engine.build_scene(env_spacing=(2.0, 2.0))

        initial_pose_tensor = torch.tensor(
            initial_pose,
            dtype=torch.float32,
            device=device,
        )
        expected_shape = (self._robot.n_qs,)
        if initial_pose_tensor.shape != expected_shape:
            raise ValueError(f"initial_pose must have shape {expected_shape}, got {tuple(initial_pose_tensor.shape)}")
        batched_initial_pose = initial_pose_tensor.expand(engine.environment_count, -1).contiguous()
        self._robot.set_default_pose(batched_initial_pose)
        self._engine.register_initial_pose()

        self._task.initialize(
            robot=self._robot,
            simulation_steps_per_control_step=self._simulation_steps_per_control_step,
        )

        self._schema = self._compute_schema()
        self._maximum_episode_steps = maximum_episode_steps
        self._episode_step_count = torch.zeros(self.count, device=device, dtype=torch.int64)
        # Starts at the default pose
        self._previous_action = self._robot.default_control.expand(self.count, -1).clone()
        # self._task.capture(reset=True)
        self._push_configuration = push_configuration
        self._push_interval_steps = 0
        if push_configuration is not None:
            control_time_step = engine.time_step * self._simulation_steps_per_control_step
            self._push_interval_steps = math.ceil(push_configuration.interval_seconds / control_time_step)
            self._linear_push_half_widths = torch.tensor(push_configuration.linear_velocity_half_width, dtype=gs.tc_float, device=device)
            self._angular_push_half_widths = torch.tensor(push_configuration.angular_velocity_half_width, dtype=gs.tc_float, device=device)
        self._steps_until_push = self._push_interval_steps

    @property
    def device(self) -> torch.device:
        return self._engine.device

    @property
    def count(self) -> int:
        return self._engine.environment_count

    @property
    def schema(self) -> EnvironmentSchema:
        return self._schema

    @torch.no_grad()
    def reset(self) -> TensorDictBase:
        self._episode_step_count.zero_()
        self._steps_until_push = self._push_interval_steps
        self._previous_action.copy_(self._robot.default_control)

        self._engine.reset()
        self._task.reset(environment_indices=torch.arange(self._engine.environment_count, device=self.device, dtype=torch.int64))
        self._engine.reset_recording_camera()
        return self._get_observation()

    @torch.no_grad()
    def reset_done(self, done: torch.Tensor) -> TensorDictBase:
        environment_indices = done.reshape(-1).nonzero().reshape(-1)

        if environment_indices.numel() > 0:
            # TODO need to profile to see if this is a bottleneck
            # For tracker, calling the engine reset might not be needed.
            self._engine.reset(environment_indices=environment_indices)
            self._task.reset(environment_indices)

            self._engine.reset_recording_camera(environment_indices)
            self._episode_step_count.masked_fill_(done, 0)
            self._previous_action[environment_indices] = self._robot.default_control
        return self._get_observation()

    @torch.no_grad()
    def step(self, action: TensorDictBase) -> tuple[TensorDictBase, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        current_action = action["control"].detach()
        self._apply_push()
        self._robot.apply_control(current_action)

        for i in range(self._simulation_steps_per_control_step):
            self._engine.update_actuators()
            self._task.step(is_control_step=i == 0)
            self._engine.step()

        self._task.measure_state()
        self._episode_step_count.add_(1)
        step_evaluation = self._task.evaluate_step(
            current_action=current_action,
            previous_action=self._previous_action,
        )

        truncated = self._episode_step_count >= self._maximum_episode_steps
        self._previous_action.copy_(current_action)

        return self._get_observation(), step_evaluation.reward, step_evaluation.terminal, truncated, step_evaluation.transition_metrics

    def _apply_push(self) -> None:
        if self._push_configuration is None:
            return
        self._steps_until_push -= 1
        if self._steps_until_push > 0:
            return
        linear_velocity_offset = self._linear_push_half_widths.new_empty((self.count, 3)).uniform_(-1.0, 1.0)
        linear_velocity_offset.mul_(self._linear_push_half_widths)
        angular_velocity_offset = self._angular_push_half_widths.new_empty((self.count, 3)).uniform_(-1.0, 1.0)
        angular_velocity_offset.mul_(self._angular_push_half_widths)
        self._robot.add_root_velocity(
            linear_velocity_offset=linear_velocity_offset,
            angular_velocity_offset=angular_velocity_offset,
        )
        self._steps_until_push = self._push_interval_steps

    def _compute_schema(self) -> EnvironmentSchema:
        observations = self._task.get_schema()

        return EnvironmentSchema(
            observations=observations,
            actions={"control": self._get_control_schema()},
        )

    def _get_control_schema(self) -> TensorSchema:
        return TensorSchema(
            shape=(self._robot.n_controlled_dofs,),
            data_type=torch.float32,
            bounds=self._robot.control_bounds,
            default_value=self._robot.default_control,
            action_scale=self._robot.control_action_scale,
        )

    def _get_observation(self) -> TensorDictBase:
        observation = self._task.observation(previous_action=self._previous_action)
        first_tensor = next(iter(observation.values()))
        return TensorDict(observation, batch_size=first_tensor.shape[:-1], device=first_tensor.device)
