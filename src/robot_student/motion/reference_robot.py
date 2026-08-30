import torch

from robot_student.engine.kinematic_robot import RobotState
from robot_student.motion.motion_library import MotionLibrary


class ReferenceRobot:
    def __init__(
        self,
        count: int,
        motion_library: MotionLibrary,
        timestep: float,
        device: torch.device | str | None = None,
    ) -> None:
        self._motion_library = motion_library
        self._timestep = timestep

        self._motion_ids = torch.empty(count, dtype=torch.int64, device=device)
        self._motion_times = torch.empty(count, dtype=torch.float32, device=device)

    def reset(self, environment_indices: torch.Tensor | None = None) -> RobotState:
        reset_count = self._motion_ids.shape[0] if environment_indices is None else environment_indices.numel()
        sampled_motion_ids, sampled_motion_times = self._motion_library.sample(reset_count)
        if environment_indices is None:
            self._motion_ids.copy_(sampled_motion_ids)
            self._motion_times.copy_(sampled_motion_times)
        else:
            self._motion_ids[environment_indices] = sampled_motion_ids
            self._motion_times[environment_indices] = sampled_motion_times

        return self._motion_library.get_state(sampled_motion_ids, sampled_motion_times)

    def get_state(self, episode_step_count: torch.Tensor) -> tuple[RobotState, torch.Tensor]:
        motion_times = self._motion_times + episode_step_count * self._timestep
        motion_finished = motion_times >= self._motion_library.motion_durations[self._motion_ids]
        return self._motion_library.get_state(self._motion_ids, motion_times), motion_finished

    def get_target_states(self, episode_step_count, target_steps):
        motion_times = self._motion_times + episode_step_count * self._timestep
        motion_times = motion_times[:, None] + target_steps * self._timestep
        motion_ids = self._motion_ids[:, None].expand_as(motion_times)

        return self._motion_library.get_state(motion_ids, motion_times)
