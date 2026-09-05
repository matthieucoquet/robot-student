import torch

from robot_student.engine.kinematic_robot import KinematicRobot, RobotState
from robot_student.motion.motion_library import MotionLibrary


class ReferenceRobot:
    def __init__(
        self,
        count: int,
        motion_library: MotionLibrary,
        timestep: float,
        device: torch.device | str | None = None,
        kinematic_robot: KinematicRobot | None = None,
    ) -> None:
        self._motion_library = motion_library
        self._timestep = timestep
        self._kinematic_robot = kinematic_robot

        self._motion_ids = torch.empty(count, dtype=torch.int64, device=device)
        self._motion_times = torch.empty(count, dtype=torch.float32, device=device)

        self._step_count = torch.zeros(count, device=device, dtype=torch.int64)

    def reset(self, *, random_sampling: bool, environment_indices: torch.Tensor | None = None) -> RobotState:
        reset_count = self._motion_ids.shape[0] if environment_indices is None else environment_indices.numel()
        if random_sampling:
            sampled_motion_ids, sampled_motion_times = self._motion_library.sample(reset_count)
        else:
            sampled_motion_ids = torch.zeros(reset_count, dtype=torch.int64, device=self._motion_ids.device)
            sampled_motion_times = torch.zeros(reset_count, dtype=torch.float32, device=self._motion_times.device)

        if environment_indices is None:
            self._step_count.zero_()
            self._motion_ids.copy_(sampled_motion_ids)
            self._motion_times.copy_(sampled_motion_times)
        else:
            self._step_count.masked_fill_(environment_indices, 0)
            self._motion_ids[environment_indices] = sampled_motion_ids
            self._motion_times[environment_indices] = sampled_motion_times

        state = self._motion_library.get_state(sampled_motion_ids, sampled_motion_times)
        self._update_display(state, environment_indices)
        return state

    def step(self, steps: int) -> tuple[RobotState, torch.Tensor]:
        self._step_count.add_(steps)
        motion_times = self._motion_times + self._step_count * self._timestep
        motion_finished = motion_times >= self._motion_library.motion_durations[self._motion_ids]
        state = self._motion_library.get_state(self._motion_ids, motion_times)
        self._update_display(state)
        return state, motion_finished

    def get_target_states(self, steps_per_control_step: int, target_steps: torch.Tensor) -> RobotState:
        motion_times = self._motion_times + self._step_count * self._timestep
        motion_times = motion_times[:, None] + target_steps * self._timestep * steps_per_control_step
        motion_ids = self._motion_ids[:, None].expand_as(motion_times)

        return self._motion_library.get_state(motion_ids, motion_times)

    def _update_display(self, state: RobotState, environment_indices: torch.Tensor | None = None) -> None:
        if self._kinematic_robot is not None:
            self._kinematic_robot.set_state(state, environment_indices=environment_indices)
