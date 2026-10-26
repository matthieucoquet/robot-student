import math
from collections.abc import Sequence

import torch
from genesis.utils.geom import inv_quat, transform_by_quat, transform_quat_by_quat
from tensordict import TensorClass

from robot_student.engine.robot import Robot
from robot_student.environment.schema import TensorSchema
from robot_student.environment.sensors import IMUSensor, Sensor
from robot_student.environment.task.task import Task, TaskEvaluation
from robot_student.util.geometry import heading_angle, quat_to_rot6d


class SimulatorObservation(TensorClass["autocast"]):
    root_position: torch.Tensor
    root_rotation: torch.Tensor
    root_velocity: torch.Tensor
    encoder_positions: torch.Tensor
    encoder_velocities: torch.Tensor
    imu_position: torch.Tensor
    imu_rotation: torch.Tensor
    imu_angular_velocity: torch.Tensor
    world_link_positions: torch.Tensor
    world_link_rotations: torch.Tensor


class RunInDirectionTask(Task):
    """Walking observations and pose costs use servo encoder coordinates, including gear play.

    Pose costs average squared errors scaled by joint weights; unspecified joints use 1.0.
    Termination checks the configured link's world height; the height reward uses the root.
    Action smoothing penalizes the mean squared change in targets normalized by the robot's control action scale.
    """

    def __init__(
        self,
        *,
        imu_link_name: str,
        termination_height_link_name: str,
        default_encoder_positions: Sequence[float],
        device: torch.device,
        key_link_names: Sequence[str] = (),
        direction: tuple[float, float] = (1.0, 0.0),
        target_speed: float = 1.0,
        target_speed_weight: float = 1.0,
        target_height: float = 0.75,
        target_height_weight: float = 1.0,
        facing_direction_weight: float = 0.5,
        control_cost_weight: float = 0.5,
        pose_cost_weight: float = 0.1,
        pose_joint_weights: dict[str, float] | None = None,
        action_smoothing_cost_weight: float = 0.0,
        termination_height_range: tuple[float, float] = (0.2, 1.0),
        encoder_positions_sensor: Sensor | None = None,
        encoder_velocities_sensor: Sensor | None = None,
        imu_sensor: IMUSensor | None = None,
    ) -> None:
        direction_norm = math.hypot(*direction)
        if not math.isclose(direction_norm, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"direction must be normalized, got norm {direction_norm}")

        self._pose_joint_weights_by_name = dict(pose_joint_weights or {})
        for joint_name, weight in self._pose_joint_weights_by_name.items():
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError(f"pose_joint_weights[{joint_name!r}] must be finite and nonnegative, got {weight}")

        self._encoder_positions_sensor = encoder_positions_sensor or Sensor()
        self._encoder_velocities_sensor = encoder_velocities_sensor or Sensor()
        self._imu_sensor = imu_sensor or IMUSensor()
        self._sensors_enabled = not (
            self._encoder_positions_sensor.is_passthrough
            and self._encoder_velocities_sensor.is_passthrough
            and self._imu_sensor.is_passthrough
        )
        self._imu_link_name = imu_link_name
        self._termination_height_link_name = termination_height_link_name
        self._key_link_names = key_link_names
        self._world_gravity = torch.tensor((0.0, 0.0, -1.0), device=device, dtype=torch.float32)
        self._direction = torch.tensor(direction, device=device, dtype=torch.float32)
        self._direction_heading = math.atan2(direction[1], direction[0])
        self._target_velocity = self._direction * target_speed
        self._target_speed_weight = target_speed_weight
        self._target_height = target_height
        self._target_height_weight = target_height_weight
        self._facing_direction_weight = facing_direction_weight
        self._control_cost_weight = control_cost_weight
        self._pose_cost_weight = pose_cost_weight
        self._action_smoothing_cost_weight = action_smoothing_cost_weight
        self._default_encoder_positions = torch.tensor(default_encoder_positions, device=device, dtype=torch.float32)
        self._minimum_healthy_height, self._maximum_healthy_height = termination_height_range

    def get_schema(self) -> dict[str, TensorSchema]:
        joint_count = self._robot.n_controlled_dofs
        link_count = self._key_link_indices.numel()

        imu_states = 6
        joint_size = 2 * joint_count
        link_size = 9 * link_count
        previous_action_size = self._robot.n_controlled_dofs
        actor_size = imu_states + joint_size + previous_action_size
        critic_size = actor_size + link_size

        sizes = {
            "actor": actor_size,
            "critic": critic_size,
        }
        return {key: TensorSchema(shape=(size,), data_type=torch.float32) for key, size in sizes.items()}

    def initialize(
        self,
        *,
        robot: Robot,
        simulation_steps_per_control_step: int,
    ) -> None:
        self._robot = robot
        if self._default_encoder_positions.shape != (robot.n_controlled_dofs,):
            raise ValueError("default_encoder_positions must contain one value per servo in control order")
        joint_names = robot.servo_joints.joint_names
        unknown_joint_names = self._pose_joint_weights_by_name.keys() - set(joint_names)
        if unknown_joint_names:
            raise ValueError(f"pose_joint_weights contains unknown servo joints: {', '.join(sorted(unknown_joint_names))}")
        self._pose_joint_weights = self._default_encoder_positions.new_tensor(
            [self._pose_joint_weights_by_name.get(joint_name, 1.0) for joint_name in joint_names]
        )
        key_link_indices = robot.get_link_indices(self._key_link_names)
        self._imu_link_index = robot.get_link_indices((self._imu_link_name,))[0]
        termination_height_link_index = robot.get_link_indices((self._termination_height_link_name,))[0]
        self._capture_link_indices = tuple(dict.fromkeys((*key_link_indices, self._imu_link_index, termination_height_link_index)))
        self._imu_capture_link_index = self._capture_link_indices.index(self._imu_link_index)
        self._termination_capture_link_index = self._capture_link_indices.index(termination_height_link_index)
        self._key_link_indices = torch.tensor(
            [self._capture_link_indices.index(index) for index in key_link_indices],
            dtype=torch.int64,
            device=robot.default_control.device,
        )
        self._simulator_observation = self._read_simulator_observation()
        self._encoder_positions_sensor.initialize(self._simulator_observation.encoder_positions)
        self._encoder_velocities_sensor.initialize(self._simulator_observation.encoder_velocities)
        if not self._imu_sensor.is_passthrough:
            self._imu_sensor.initialize(self._simulator_observation.imu_rotation, self._simulator_observation.imu_angular_velocity)

    @torch.no_grad()
    def reset(self, environment_indices: torch.Tensor) -> None:
        simulator_observation = self._read_simulator_observation(environment_indices)
        self._simulator_observation[environment_indices] = simulator_observation
        if not self._encoder_positions_sensor.is_passthrough:
            self._encoder_positions_sensor.reset(self._simulator_observation.encoder_positions, environment_indices)
        if not self._encoder_velocities_sensor.is_passthrough:
            self._encoder_velocities_sensor.reset(self._simulator_observation.encoder_velocities, environment_indices)
        if not self._imu_sensor.is_passthrough:
            self._imu_sensor.reset(
                self._simulator_observation.imu_rotation,
                self._simulator_observation.imu_angular_velocity,
                environment_indices,
            )

    def step(self, is_control_step: bool) -> None:
        if is_control_step:
            # Get forces just before the physics step are executed
            self._normalized_control_forces = self._robot.get_normalized_control_forces()

    @torch.no_grad()
    def measure_state(self) -> None:
        """Capture the full batch and advance sensor histories once after a control interval."""
        self._simulator_observation = self._read_simulator_observation()
        self._encoder_positions_sensor.update(self._simulator_observation.encoder_positions)
        self._encoder_velocities_sensor.update(self._simulator_observation.encoder_velocities)
        if not self._imu_sensor.is_passthrough:
            self._imu_sensor.update(self._simulator_observation.imu_rotation, self._simulator_observation.imu_angular_velocity)

    @torch.no_grad()
    def _read_simulator_observation(self, environment_indices: torch.Tensor | None = None) -> SimulatorObservation:
        positions = self._robot.get_generalized_positions(environment_indices)
        velocities = self._robot.get_generalized_velocities(environment_indices)
        link_positions = self._robot.get_links_position(environment_indices, link_indices=self._capture_link_indices)
        link_rotations = self._robot.get_links_rotation(environment_indices, link_indices=self._capture_link_indices)

        return SimulatorObservation(
            root_position=positions[..., :3],
            root_rotation=positions[..., 3 : self._robot.n_root_qs],
            root_velocity=velocities[..., :3],
            encoder_positions=self._robot.servo_joints.get_encoder_positions(environment_indices),
            encoder_velocities=self._robot.servo_joints.get_encoder_velocities(environment_indices),
            imu_position=link_positions[..., self._imu_capture_link_index, :],
            imu_rotation=link_rotations[..., self._imu_capture_link_index, :],
            imu_angular_velocity=self._robot.get_links_angular_velocity(environment_indices, link_indices=(self._imu_link_index,))[
                ..., 0, :
            ],
            world_link_positions=link_positions,
            world_link_rotations=link_rotations,
            batch_size=positions.shape[0],
        )

    def observation(self, *, previous_action: torch.Tensor) -> dict[str, torch.Tensor]:
        simulator_observation = self._simulator_observation
        critic_inverse_imu_rotation = inv_quat(simulator_observation.imu_rotation)
        imu_angular_velocity = transform_by_quat(simulator_observation.imu_angular_velocity, critic_inverse_imu_rotation)
        projected_gravity = transform_by_quat(self._world_gravity, critic_inverse_imu_rotation)
        encoder_positions = simulator_observation.encoder_positions - self._robot.default_encoder_positions
        encoder_velocities = simulator_observation.encoder_velocities
        critic_components = (imu_angular_velocity, projected_gravity, encoder_positions, encoder_velocities, previous_action)
        actor_components = critic_components
        if self._sensors_enabled:
            if not self._imu_sensor.is_passthrough:
                imu_angular_velocity = self._imu_sensor.angular_velocity
                projected_gravity = self._imu_sensor.projected_gravity
            if not self._encoder_positions_sensor.is_passthrough:
                encoder_positions = self._encoder_positions_sensor.observation - self._robot.default_encoder_positions
            if not self._encoder_velocities_sensor.is_passthrough:
                encoder_velocities = self._encoder_velocities_sensor.observation
            actor_components = (imu_angular_velocity, projected_gravity, encoder_positions, encoder_velocities, previous_action)
        if self._key_link_indices.numel():
            world_link_positions = simulator_observation.world_link_positions.index_select(-2, self._key_link_indices)
            world_link_rotations = simulator_observation.world_link_rotations.index_select(-2, self._key_link_indices)
            link_positions = transform_by_quat(
                world_link_positions - simulator_observation.imu_position.unsqueeze(-2), critic_inverse_imu_rotation.unsqueeze(-2)
            ).flatten(start_dim=-2)
            link_rotations = quat_to_rot6d(transform_quat_by_quat(world_link_rotations, critic_inverse_imu_rotation.unsqueeze(-2))).flatten(
                start_dim=-2
            )
            critic_components = (*critic_components, link_positions, link_rotations)
        return {
            "actor": torch.cat(actor_components, dim=-1),
            "critic": torch.cat(critic_components, dim=-1),
        }

    def evaluate_step(
        self,
        *,
        current_action: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> TaskEvaluation:
        """Report raw reward terms and signed, weighted contributions under task/reward_contribution/."""
        simulator_observation = self._simulator_observation
        root_height = simulator_observation.root_position[..., 2]
        termination_height = simulator_observation.world_link_positions[..., self._termination_capture_link_index, 2]
        termination_height_is_healthy = termination_height >= self._minimum_healthy_height
        termination_height_is_healthy.logical_and_(termination_height <= self._maximum_healthy_height)
        terminal = ~termination_height_is_healthy

        planar_velocity = simulator_observation.root_velocity[..., :2]
        target_velocity_error = torch.linalg.vector_norm(planar_velocity - self._target_velocity, dim=-1)
        target_velocity_reward = target_velocity_error.mul(12.0).neg_().exp_()

        target_height_error = root_height - self._target_height
        target_height_reward = target_height_error.clamp(max=0.0).abs_().mul_(25.0).neg_().exp_()

        facing_direction_reward = torch.cos(heading_angle(simulator_observation.root_rotation) - self._direction_heading)

        control_cost = torch.mean(self._normalized_control_forces.square(), dim=-1)
        pose_cost = torch.mean(
            (simulator_observation.encoder_positions - self._default_encoder_positions).square().mul_(self._pose_joint_weights), dim=-1
        )
        action_change = (current_action - previous_action) / self._robot.control_action_scale
        action_smoothing_cost = torch.mean(action_change.square(), dim=-1)
        stay_alive_reward = termination_height_is_healthy * 0.05
        reward = (
            stay_alive_reward
            + self._target_speed_weight * target_velocity_reward
            + self._target_height_weight * target_height_reward
            + self._facing_direction_weight * facing_direction_reward
            - self._control_cost_weight * control_cost
            - self._pose_cost_weight * pose_cost
            - self._action_smoothing_cost_weight * action_smoothing_cost
        )

        stay_alive_reward_mean = stay_alive_reward.mean()
        target_velocity_reward_mean = target_velocity_reward.mean()
        target_height_reward_mean = target_height_reward.mean()
        facing_direction_reward_mean = facing_direction_reward.mean()
        control_cost_mean = control_cost.mean()
        pose_cost_mean = pose_cost.mean()
        action_smoothing_cost_mean = action_smoothing_cost.mean()

        return TaskEvaluation(
            reward=reward,
            terminal=terminal,
            transition_metrics={
                "task/target_velocity_error_mean": target_velocity_error.mean(),
                "task/target_height_error_mean": target_height_error.abs().mean(),
                "task/stay_alive_reward_mean": stay_alive_reward_mean,
                "task/target_velocity_reward_mean": target_velocity_reward_mean,
                "task/target_height_reward_mean": target_height_reward_mean,
                "task/facing_direction_reward_mean": facing_direction_reward_mean,
                "task/root_height_mean": root_height.mean(),
                "task/termination_height_mean": termination_height.mean(),
                "task/control_cost_mean": control_cost_mean,
                "task/pose_cost_mean": pose_cost_mean,
                "task/action_smoothing_cost_mean": action_smoothing_cost_mean,
                "task/reward_mean": reward.mean(),
            },
        )
