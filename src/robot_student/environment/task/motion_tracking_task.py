from dataclasses import dataclass
from pathlib import Path

import torch
from genesis.utils.geom import transform_quat_by_quat, xyz_to_quat

from robot_student.engine.genesis_engine import GenesisEngine
from robot_student.engine.robot import Robot, scale_joint_position_limits
from robot_student.engine.robot_state import GeneralizedRobotState, RobotState
from robot_student.environment.task.task import Task
from robot_student.motion import MotionLibrary, ReferenceRobot


@dataclass(frozen=True, kw_only=True, slots=True)
class ResetPerturbationConfiguration:
    root_position_half_width: tuple[float, float, float]  # World x, y, z in meters.
    root_rotation_half_width: tuple[float, float, float]  # World-frame roll, pitch, yaw in radians.
    root_linear_velocity_half_width: tuple[float, float, float]  # World x, y, z in meters per second.
    root_angular_velocity_half_width: tuple[float, float, float]  # World x, y, z in radians per second.
    joint_position_half_width: float  # Radians, applied independently to each joint.


class MotionTrackingTask(Task):
    def __init__(
        self,
        xml_path: Path,
        motion_library: MotionLibrary,
        show_reference_motion: bool = False,
        reference_motion_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        soft_joint_position_limit_factor: float | None = None,
        reset_perturbation_configuration: ResetPerturbationConfiguration | None = None,
    ) -> None:
        super().__init__()
        self._xml_path = xml_path
        self._motion_library = motion_library
        self._reference_motion_offset = reference_motion_offset
        self._show_reference_motion = show_reference_motion
        self._soft_joint_position_limit_factor = soft_joint_position_limit_factor
        self._reset_perturbation_configuration = reset_perturbation_configuration

    def setup_scene(self, engine: GenesisEngine) -> None:
        self._kinematic_robot = None
        if self._show_reference_motion:
            self._kinematic_robot = engine.add_kinematic_robot(
                self._xml_path,
                position_offset=(0.0, 0.0, 0.0),
                color=(0.15, 0.55, 1.0, 0.7),
                name="reference_robot",
            )

        self._reference_robot = ReferenceRobot(
            engine.environment_count,
            self._motion_library,
            engine.time_step,
            engine.device,
            kinematic_robot=self._kinematic_robot,
            display_offset=self._reference_motion_offset,
        )
        self._reference_state = self._motion_library.get_state(
            torch.zeros(engine.environment_count, dtype=torch.int64, device=engine.device),
            torch.zeros(engine.environment_count, dtype=torch.float32, device=engine.device),
        )
        self._motion_finished = torch.zeros(engine.environment_count, dtype=torch.bool, device=engine.device)

    def initialize(
        self,
        *,
        robot: Robot,
        key_link_indices: torch.Tensor,
        simulation_steps_per_control_step: int,
        global_observation: bool,
    ) -> None:
        self._robot = robot
        self._key_link_indices = key_link_indices
        self._simulation_steps_per_control_step = simulation_steps_per_control_step
        self._global_observation = global_observation
        lower_bounds, upper_bounds = self._robot.get_joint_dof_limits()
        self._soft_joint_position_lower_bounds, self._soft_joint_position_upper_bounds = scale_joint_position_limits(
            lower_bounds, upper_bounds, self._soft_joint_position_limit_factor
        )

    def reset(self, environment_indices: torch.Tensor) -> None:
        reference_state = self._reference_robot.reset(environment_indices=environment_indices)
        reset_state = self._apply_reset_perturbations(reference_state)
        self._robot.set_state(reset_state, environment_indices=environment_indices)
        self._reference_state.copy_environments_(environment_indices, reference_state)
        self._motion_finished.index_fill_(0, environment_indices, False)

    def _apply_reset_perturbations(self, reference_state: RobotState) -> GeneralizedRobotState:
        configuration = self._reset_perturbation_configuration
        if configuration is None:
            return reference_state

        def sample_noise(value: torch.Tensor, half_width: tuple[float, float, float] | float) -> torch.Tensor:
            return torch.empty_like(value).uniform_(-1.0, 1.0).mul_(value.new_tensor(half_width))

        rotation_angles = sample_noise(reference_state.root_rotation[..., 1:], configuration.root_rotation_half_width)
        rotation_offset = xyz_to_quat(rotation_angles, rpy=True)
        joint_positions = sample_noise(reference_state.joint_dof_positions, configuration.joint_position_half_width)
        joint_positions.add_(reference_state.joint_dof_positions).clamp_(
            min=self._soft_joint_position_lower_bounds,
            max=self._soft_joint_position_upper_bounds,
        )
        # set_state recomputes link states from these perturbed generalized coordinates.
        return GeneralizedRobotState(
            root_position=sample_noise(reference_state.root_position, configuration.root_position_half_width).add_(
                reference_state.root_position
            ),
            root_rotation=transform_quat_by_quat(reference_state.root_rotation, rotation_offset),
            joint_dof_positions=joint_positions,
            root_velocity=sample_noise(reference_state.root_velocity, configuration.root_linear_velocity_half_width).add_(
                reference_state.root_velocity
            ),
            root_angular_velocity=sample_noise(reference_state.root_angular_velocity, configuration.root_angular_velocity_half_width).add_(
                reference_state.root_angular_velocity
            ),
            joint_dof_velocities=reference_state.joint_dof_velocities,
            batch_size=reference_state.batch_size,
        )

    def step(self, is_control_step: bool) -> None:
        if self._show_reference_motion:
            self._reference_state, self._motion_finished = self._reference_robot.step(1)
        elif is_control_step:
            self._reference_state, self._motion_finished = self._reference_robot.step(self._simulation_steps_per_control_step)
