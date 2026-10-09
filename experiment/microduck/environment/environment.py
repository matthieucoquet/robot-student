from dataclasses import dataclass

from robot_student.engine.genesis_engine import GenesisEngine
from robot_student.engine.robot import CommandDelayConfiguration, ObservationDelayConfiguration
from robot_student.environment import RobotEnvironment, RunInDirectionTask
from robot_student.environment.environment import Environment
from robot_student.run.environment_factory import EnvironmentFactory

from .robot_configuration import microduck_configuration


@dataclass(frozen=True, kw_only=True, slots=True)
class PPOEnvironmentFactory(EnvironmentFactory):
    control_frequency: int = 50
    simulation_frequency: int = 200

    def create_environment(
        self,
        engine: GenesisEngine,
    ) -> Environment:
        mjcf_path, control_mode, initial_pose, initial_joint_positions = microduck_configuration()

        command_delay_configuration = CommandDelayConfiguration(
            delay_physics_steps_range=(3, 6),  # 15–30 ms at 200 Hz.
        )
        imu_link_name = "trunk_base"
        observation_delay_configuration = ObservationDelayConfiguration(
            joint_velocity_delay_control_steps=1,
            imu_link_name=imu_link_name,
            imu_delay_control_steps_range=(0, 1),
        )

        # robot_state_noise = NoiseConfiguration(
        #     root_velocity=UniformNoise(half_width=0.5),
        #     root_angular_velocity=UniformNoise(half_width=0.2),
        #     joint_dof_positions=UniformNoise(half_width=0.01),
        #     joint_dof_velocities=UniformNoise(half_width=0.5),
        #     world_link_positions=UniformNoise(half_width=0.25),
        #     world_link_rotations=UniformNoise(half_width=0.05),
        # )

        # reset_perturbation_configuration = ResetPerturbationConfiguration(
        #     root_position_half_width=(0.05, 0.05, 0.01),
        #     root_rotation_half_width=(0.1, 0.1, 0.2),
        #     root_linear_velocity_half_width=(0.5, 0.5, 0.2),
        #     root_angular_velocity_half_width=(0.52, 0.52, 0.78),
        #     joint_position_half_width=0.1,
        # )

        # push_configuration = PushConfiguration(
        #     interval_seconds=2.0, linear_velocity_half_width=(0.5, 0.5, 0.2), angular_velocity_half_width=(0.52, 0.52, 0.78)
        # )

        # domain_randomization_configuration = DomainRandomizationConfiguration(
        #     friction_ratio_range=(0.3, 1.6),
        #     default_joint_position_offset_range=(-0.01, 0.01),
        #     center_of_mass=CenterOfMassRandomization(
        #         link_name="torso_link",
        #         x_range=(-0.025, 0.025),
        #         y_range=(-0.05, 0.05),
        #         z_range=(-0.05, 0.05),
        #     ),
        # )

        key_link_names = ("ankle_left", "ankle_right", "jaw_soft")

        task = RunInDirectionTask(
            device=engine.device,
            imu_link_name=imu_link_name,
            default_joint_positions=initial_joint_positions,
            height_range=(0.06, 0.2),
            target_height=0.11,
            target_speed=0.4,
            target_speed_weight=1.5,
            target_height_weight=0.0,
            facing_direction_weight=0.05,
            control_cost_weight=0.1,
            pose_cost_weight=0.05,
        )

        return RobotEnvironment(
            engine,
            mjcf_path,
            control_mode=control_mode,
            task=task,
            control_frequency=self.control_frequency,
            initial_pose=initial_pose,
            key_link_names=key_link_names,
            command_delay_configuration=command_delay_configuration,
            observation_delay_configuration=observation_delay_configuration,
            # push_configuration=push_configuration,
        )
