import torch
from robot_student.motion.motion_library import MotionLibrary


class ReferenceRobot:
    def __init__(
        self,
        count: int,
        motion_library: MotionLibrary,
        device: torch.device | str | None = None,
    ) -> None:
        self._motion_library = motion_library

        self._motion_ids = torch.empty(count, dtype=torch.int64, device=device)
        self._motion_times = torch.empty(count, dtype=torch.float32, device=device)


    def reset(self, environment_indices: torch.Tensor | None = None) -> None:
        reset_count = environment_indices.numel()
        sampled_motion_ids, sampled_motion_times = self._motion_library.sample(reset_count)
        if environment_indices is None:
            self._motion_ids.copy_(sampled_motion_ids)
            self._motion_times.copy_(sampled_motion_times)
        else:
            self._motion_ids[environment_indices] = sampled_motion_ids
            self._motion_times[environment_indices] = sampled_motion_times

        state = self._motion_library.get_state(sampled_motion_ids, sampled_motion_times)

        define the state?
        same class for motion and physics/kinematic character?
        self._robot.set_state(state, environment_indices=environment_indices)

        return state
        # generalized_positions = torch.cat(
        #     (frame.root_position, frame.root_rotation, frame.joint_dof_position),
        #     dim=-1,
        # )
        # generalized_velocities = torch.cat(
        #     (frame.root_velocity, frame.root_angular_velocity, frame.joint_dof_velocities),
        #     dim=-1,
        # )
        # self._robot.set_generalized_state(
        #     generalized_positions,
        #     generalized_velocities,
        #     environment_indices=environment_indices,
        # )