from typing import Any

import torch
from genesis.utils.geom import inv_quat, inv_transform_by_quat, transform_by_quat, transform_quat_by_quat

from robot_student.engine.genesis_engine import GenesisEngine
from robot_student.engine.kinematic_robot import RobotState
from robot_student.engine.robot import Robot
from robot_student.environment.schema import TensorSchema
from robot_student.environment.task.motion_tracking_task import MotionTrackingTask
from robot_student.environment.task.task import TaskFeedback
from robot_student.util.geometry import quat_to_rot6d


class BeyondMimicTask(MotionTrackingTask):
    def __init__(
        self,
        engine: GenesisEngine,
        # environment_count: int,
        # xml_path: Path,
        # motion_library: MotionLibrary,
        # target_steps: Sequence[float],
        # joint_reward_weight: Sequence[float],
        # random_reference_sampling: bool = False,
        # show_reference_motion: bool = False,
        # reference_motion_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        anchor_link_name: str = "pelvis",
    ) -> None:
        super().__init__()
        self._anchor_link_name = anchor_link_name

    def setup_scene(self, engine: GenesisEngine) -> None:
        pass

    def initialize(
        self,
        *,
        robot: Robot,
        key_link_indices: torch.Tensor,
        simulation_steps_per_control_step: int,
        global_observation: bool,
    ) -> None:
        super().initialize(
            robot=robot,
            key_link_indices=key_link_indices,
            simulation_steps_per_control_step=simulation_steps_per_control_step,
            global_observation=global_observation,
        )
        self._anchor_link_index = self._robot.get_link_indices([self._anchor_link_name])[0]

    def reset(self, environment_indices: torch.Tensor) -> None:
        self._reference_state = something

    def get_schema(self) -> dict[str, TensorSchema]:
        pass

    def observation(self, robot_state: RobotState, *, previous_action: torch.Tensor) -> dict[str, torch.Tensor]:
        motion_phase = torch.cat(
            (self._reference_state.joint_dof_positions, self._reference_state.joint_dof_velocities),
            dim=-1,
        )

        anchor_position = robot_state.world_link_positions[..., self._anchor_link_index, :]
        anchor_rotation = robot_state.world_link_rotations[..., self._anchor_link_index, :]
        inverse_anchor_rotation = inv_quat(anchor_rotation)
        reference_anchor_position = self._reference_state.world_link_positions[..., self._anchor_link_index, :]
        reference_anchor_rotation = self._reference_state.world_link_rotations[..., self._anchor_link_index, :]

        anchor_position_error = transform_by_quat(reference_anchor_position - anchor_position, inverse_anchor_rotation)
        anchor_rotation_error = quat_to_rot6d(transform_quat_by_quat(reference_anchor_rotation, inverse_anchor_rotation))

        root_linear_velocities = robot_state.root_linear_velocities
        root_angular_velocities = robot_state.root_angular_velocities

        joint_positions = robot_state.joint_dof_positions
        joint_velocities = robot_state.joint_dof_velocities

        world_link_positions = robot_state.world_link_positions
        world_link_rotations = robot_state.world_link_rotations
        link_positions = transform_by_quat(world_link_positions - anchor_position, inverse_anchor_rotation).flatten(start_dim=-2)
        link_rotations = quat_to_rot6d(transform_quat_by_quat(world_link_rotations, inverse_anchor_rotation)).flatten(start_dim=-2)

        return {
            "motion_phase": motion_phase,
            "anchor_position_error": anchor_position_error,
            "anchor_rotation_error": anchor_rotation_error,
            "root_linear_velocities": root_linear_velocities,
            "root_angular_velocities": root_angular_velocities,
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "link_positions": link_positions,
            "link_rotations": link_rotations,
            "previous_action": previous_action.clone(),
        }

    def _compute_reward(
        self,
        state: RobotState,
        reference: RobotState,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        pass

    def _compute_terminal(self, state: RobotState, reference: RobotState):
        pass

    def step(self, is_control_step: bool) -> None:
        pass

    def compute_feedback(self, state: RobotState, **kwargs: Any) -> TaskFeedback:
        pass
