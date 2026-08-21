from pathlib import Path
from typing import TYPE_CHECKING

import torch
from genesis.utils.geom import transform_by_quat, transform_quat_by_quat
from tensordict import TensorDict, TensorDictBase

from robot_student.engine import ControlMode, ReferenceRobot
from robot_student.environment.environment import CharacterEnvironment
from robot_student.environment.schema import EnvironmentSchema, TensorSchema
from robot_student.motion import MotionLibrary, ReferenceRobot
from robot_student.util.geometry import inverse_heading_rotation, quat_to_rot6d

if TYPE_CHECKING:
    from robot_student.engine.genesis_engine import GenesisEngine

type RootState = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class MotionTrackingEnvironment(CharacterEnvironment):
    def __init__(
        self,
        engine: "GenesisEngine",
        motion_library: MotionLibrary,
        xml_path: Path,
        environment_count: int,
        control_mode: ControlMode,
        task: CharacterTask,
        control_frequency: int,
        target_steps: list,
        maximum_episode_steps: int = 1_000,
    ) -> None:

        # self._motion_library = motion_library
        self._reference_robot = ReferenceRobot(motion_library, environment_count, engine.device)
        # self._target_steps = torch.tensor(target_steps, dtype=torch.float32, device=engine.device)

        super().__init__(
            engine,
            xml_path,
            environment_count,
            control_mode,
            task,
            control_frequency,
            initial_pose=None,
            maximum_episode_steps=maximum_episode_steps,
        )

    def _compute_schema(self) -> EnvironmentSchema:
        observation_type = torch.float32
        root_observation_size = 1 + 6 + 3 + 3
        proprioception_size = root_observation_size + 2 * self._robot.n_joint_dofs
        target_size = self._target_steps * (3 + 6 + self._robot.n_joint_dofs)

        return EnvironmentSchema(
            observations={
                "proprioception": TensorSchema(
                    shape=(proprioception_size,),
                    data_type=observation_type,
                ),
                "target": TensorSchema(
                    shape=(target_size,),
                    data_type=observation_type,
                ),
            },
            actions={"control": self._robot.get_action_schema()},
        )

    def reset(self) -> TensorDictBase:
        self._episode_step_count.zero_()
        reset_state = self._reset_characters()
        root_state = self._robot.get_root_state()
        return self._get_character_observation(root_state)

    def reset_done(self, done: torch.Tensor) -> TensorDictBase:
        self._episode_step_count.masked_fill_(done, 0)
        environment_indices = done.reshape(-1).nonzero().reshape(-1)

        if environment_indices.numel() == 0:
            return

        self._engine.reset(environment_indices=environment_indices)  # Probably not needed to call genesis reset?

        reset_state = self._reference_robot.reset(environment_indices)

        root_state = self._robot.get_root_state()
        return self._get_character_observation(root_state)

    # def step(self, action: TensorDictBase) -> tuple[TensorDictBase, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    #     self._robot.apply_action(action["control"].detach())

    #     for _ in range(self._simulation_steps_per_control_step):
    #         self._engine.step()

    #     root_state = self._robot.get_root_state()
    #     root_position, root_rotation, root_velocity, _ = root_state
    #     observation = self._get_character_observation(root_state)
    #     joint_positions = self._robot.get_joint_dof_positions()
    #     task_step = self._task.step(root_position, root_rotation, root_velocity, joint_positions, normalized_control_forces)

    #     self._episode_step_count.add_(1)
    #     truncated = self._episode_step_count >= self._maximum_episode_steps

    #     return observation, task_step.reward, task_step.terminal, truncated, task_step.transition_metrics

    def _get_observation(self) -> TensorDictBase:
        # root_state? Should it be one state object retrieved once?
        observation = self._get_character_observation(root_state)
        target = self._get_character_observation(root_state)
        observation.update(target)
        return observation

    def _get_target_observation(self, root_state: RootState) -> TensorDictBase:
        motion_times = self._motion_times[:, None] + self._target_steps * self._engine.get_timestep()
        motion_ids = self._motion_ids[:, None].expand_as(motion_times)

        targets = self._motion_library.get_state(motion_ids, motion_times)

        root_position, root_rotation, root_velocity, root_angular_velocity = root_state
        if self._global_observation:
            target_root_position = targets.root_position - root_position
            target_root_rotation = targets.root_rotation
        else:
            target_root_position = targets.root_position - targets.root_position[..., 0, :]

            reference_root_rotation = targets.root_rotation[..., 0, :]
            inverse_heading = inverse_heading_rotation(reference_root_rotation)
            target_root_position = transform_by_quat(target_root_position, inverse_heading)
            target_root_position[..., 2] = targets.root_position[..., 2]  # We keep the height

            target_root_rotation = transform_quat_by_quat(targets.root_rotation, inverse_heading)

        target_root_rotation = quat_to_rot6d(target_root_rotation)

        target_components = [
            target_root_position,
            target_root_rotation,
            targets.joint_dof_position,  # same as character mimickit use 6D for each joint, for now we use 1D
        ]
        target = torch.cat(target_components, dim=-1)
        return TensorDict({"target": target}, batch_size=target.shape[:-1], device=target.device)
