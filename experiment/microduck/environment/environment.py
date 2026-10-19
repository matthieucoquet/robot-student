import math
from dataclasses import dataclass

from robot_student.engine.genesis_engine import GenesisEngine
from robot_student.engine.robot import (
    CenterOfMassRandomization,
    CommandDelayConfiguration,
    DomainRandomizationConfiguration,
    MassInertiaRandomization,
)
from robot_student.environment import RobotEnvironment, RunInDirectionTask
from robot_student.environment.environment import Environment
from robot_student.environment.sensors import IMUSensor, Sensor
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
        domain_randomization_configuration = DomainRandomizationConfiguration(
            friction_ratio_range=(0.7, 1.3),
            friction_link_names=("ankle_left", "ankle_right"),
            friction_shared_across_links=True,
            center_of_mass=CenterOfMassRandomization(
                link_names=("trunk_base", "jaw_soft"),
                x_range=(-0.003, 0.003),
                y_range=(-0.003, 0.003),
                z_range=(-0.003, 0.003),
            ),
            mass_inertia=MassInertiaRandomization(link_names=("trunk_base",), scale_range=(0.95, 1.05)),
            armature_scale_range=(0.9, 1.1),
        )
        imu_link_name = "trunk_base"

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

        key_link_names = ("ankle_left", "ankle_right", "jaw_soft")

        task = RunInDirectionTask(
            device=engine.device,
            imu_link_name=imu_link_name,
            key_link_names=key_link_names,
            default_joint_positions=initial_joint_positions,
            height_range=(0.06, 0.2),
            target_height=0.11,
            target_speed=0.4,
            target_speed_weight=1.5,
            target_height_weight=0.0,
            facing_direction_weight=0.05,
            control_cost_weight=0.1,
            pose_cost_weight=0.05,
            joint_dof_positions_sensor=Sensor(bias_half_width=0.015, noise_half_width=0.001),  # Radians.
            joint_dof_velocities_sensor=Sensor(noise_half_width=0.25, delay_control_steps_range=(1, 1)),  # Radians per second.
            imu_sensor=IMUSensor(
                mounting_error_max_angle=math.radians(6.0),
                angular_velocity_noise_half_width=0.03,  # Radians per second in the IMU frame.
                projected_gravity_noise_half_width=0.01,
                delay_control_steps_range=(0, 1),
            ),
        )

        return RobotEnvironment(
            engine,
            mjcf_path,
            control_mode=control_mode,
            task=task,
            control_frequency=self.control_frequency,
            initial_pose=initial_pose,
            ground_friction=0.5,  # Below the minimum foot coefficient, since Genesis takes the maximum of both surfaces.
            domain_randomization_configuration=domain_randomization_configuration,
            command_delay_configuration=command_delay_configuration,
            # push_configuration=push_configuration,
        )
