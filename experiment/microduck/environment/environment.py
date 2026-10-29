from dataclasses import dataclass
from pathlib import Path

from robot_student.engine.genesis_engine import GenesisEngine
from robot_student.engine.robot import (
    CenterOfMassRandomization,
    DomainRandomizationConfiguration,
    MassInertiaRandomization,
)
from robot_student.environment import RobotEnvironment, RunInDirectionTask
from robot_student.environment.environment import Environment
from robot_student.environment.robot_environment import PushConfiguration
from robot_student.environment.sensors import SensorConfiguration
from robot_student.environment.task.beyond_mimic_task import BeyondMimicSensorConfiguration, BeyondMimicTask
from robot_student.environment.task.motion_tracking_task import ResetPerturbationConfiguration
from robot_student.motion import MotionLibrary, ReferenceSampling
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

        command_delay_configuration = None
        # command_delay_configuration = CommandDelayConfiguration(
        #     delay_physics_steps_range=(3, 6),  # 15–30 ms at 200 Hz.
        # )
        domain_randomization_configuration = None
        # domain_randomization_configuration = DomainRandomizationConfiguration(
        #     friction_ratio_range=(0.7, 1.3),
        #     friction_link_names=("ankle_left", "ankle_right"),
        #     friction_shared_across_links=True,
        #     center_of_mass=CenterOfMassRandomization(
        #         link_names=("trunk_base", "jaw_soft"),
        #         x_range=(-0.003, 0.003),
        #         y_range=(-0.003, 0.003),
        #         z_range=(-0.003, 0.003),
        #     ),
        #     mass_inertia=MassInertiaRandomization(link_names=("trunk_base",), scale_range=(0.95, 1.05)),
        #     armature_scale_range=(0.9, 1.1),
        # )
        imu_link_name = "trunk_base"

        # reset_perturbation_configuration = ResetPerturbationConfiguration(
        #     root_position_half_width=(0.05, 0.05, 0.01),
        #     root_rotation_half_width=(0.1, 0.1, 0.2),
        #     root_linear_velocity_half_width=(0.5, 0.5, 0.2),
        #     root_angular_velocity_half_width=(0.52, 0.52, 0.78),
        #     joint_position_half_width=0.1,
        # )

        push_configuration = None
        # push_configuration = PushConfiguration(
        #     interval_seconds=4.5,  # Midpoint of microduck_rl's 3–6 second interval range.
        #     linear_velocity_half_width=(0.3, 0.3, 0.0),
        #     angular_velocity_half_width=(0.0, 0.0, 0.0),
        # )

        key_link_names = ("ankle_left", "ankle_right", "jaw_soft")

        task = RunInDirectionTask(
            device=engine.device,
            imu_link_name=imu_link_name,
            termination_height_link_name="jaw_soft",
            key_link_names=key_link_names,
            default_servo_encoder_positions=initial_joint_positions,
            termination_height_range=(0.195, 0.27),  # jaw_soft origin is about 0.233 m high in the default pose.
            target_height=0.11,
            target_speed=0.3,
            target_speed_weight=1.5,
            target_height_weight=0.0,
            facing_direction_weight=0.05,
            control_cost_weight=0.0,
            pose_cost_weight=0.10,
            pose_joint_weights={
                "left_hip_yaw": 0.1,
                "left_hip_roll": 0.1,
                "left_hip_pitch": 0.1,
                "left_knee": 0.1,
                "left_ankle": 0.1,
                "neck_pitch": 10.0,
                "head_pitch": 10.0,
                "head_yaw": 10.0,
                "head_roll": 10.0,
                "right_hip_yaw": 0.1,
                "right_hip_roll": 0.1,
                "right_hip_pitch": 0.1,
                "right_knee": 0.1,
                "right_ankle": 0.1,
            },
            action_smoothing_cost_weight=0.1,
            # servo_encoder_positions_sensor=Sensor(bias_half_width=0.015, noise_half_width=0.001),  # Radians.
            # servo_encoder_velocities_sensor=Sensor(noise_half_width=0.25, delay_control_steps_range=(1, 1)),  # Radians per second.
            # imu_sensor=IMUSensor(
            #     mounting_error_max_angle=math.radians(6.0),
            #     angular_velocity_noise_half_width=0.03,  # Radians per second in the IMU frame.
            #     projected_gravity_noise_half_width=0.01,
            #     delay_control_steps_range=(0, 1),
            # ),
        )

        return RobotEnvironment(
            engine,
            xml_path=mjcf_path,
            control_mode=control_mode,
            task=task,
            control_frequency=self.control_frequency,
            initial_pose=initial_pose,
            # ground_friction=0.5,  # Below the minimum foot coefficient, since Genesis takes the maximum of both surfaces.
            domain_randomization_configuration=domain_randomization_configuration,
            command_delay_configuration=command_delay_configuration,
            push_configuration=push_configuration,
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class BeyondMimicEnvironmentFactory(EnvironmentFactory):
    reference_sampling: ReferenceSampling = ReferenceSampling.ADAPTIVE
    show_reference_motion: bool = False
    reference_motion_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    control_frequency: int = 50
    simulation_frequency: int = 200
    enable_randomization: bool = True

    def create_environment(self, engine: GenesisEngine) -> Environment:

        mjcf_path, control_mode, initial_pose, _ = microduck_configuration()
        control_mode.action_limit_scale = None

        experiment_path = Path(__file__).parent.parent
        motion_path = experiment_path / "dataset" / "preprocessed" / "BG_Normal_Walking_00001.pt"
        if not motion_path.is_file():
            raise FileNotFoundError(f"Preprocessed motion not found: {motion_path}")

        motion_library = MotionLibrary([motion_path], device=engine.device, reference_sampling=self.reference_sampling)

        key_link_names = (
            "trunk_base",
            "hip_l",
            "upper_leg_left",
            "leg",
            "ankle_left",
            "hip_l_2",
            "upper_leg_right",
            "leg_2",
            "ankle_right",
            "neck",
            "neck_pitch",
            "yaw_roll_motion",
            "jaw_soft",
        )

        command_delay_configuration = None
        # command_delay_configuration = CommandDelayConfiguration(
        #     delay_physics_steps_range=(3, 6),  # 15–30 ms at 200 Hz.
        # )

        sensor_configuration = BeyondMimicSensorConfiguration(
            root_velocity=SensorConfiguration(noise_half_width=0.075),
            root_angular_velocity=SensorConfiguration(noise_half_width=0.2),
            servo_encoder_positions=SensorConfiguration(noise_half_width=0.01),
            servo_encoder_velocities=SensorConfiguration(noise_half_width=0.5),
            anchor_position=SensorConfiguration(noise_half_width=0.0375),
            anchor_rotation=SensorConfiguration(noise_half_width=0.05),
        )
        reset_perturbation_configuration = ResetPerturbationConfiguration(
            root_position_half_width=(0.0075, 0.0075, 0.0015),
            root_rotation_half_width=(0.1, 0.1, 0.2),
            root_linear_velocity_half_width=(0.075, 0.075, 0.03),
            root_angular_velocity_half_width=(0.52, 0.52, 0.78),
            joint_position_half_width=0.1,
        )

        push_configuration = PushConfiguration(
            interval_seconds=2.0,
            linear_velocity_half_width=(0.075, 0.075, 0.03),
            angular_velocity_half_width=(0.52, 0.52, 0.78),
        )
        # push_configuration = PushConfiguration(
        #     interval_seconds=4.5,  # Midpoint of microduck_rl's 3–6 second interval range.
        #     linear_velocity_half_width=(0.3, 0.3, 0.0),
        #     angular_velocity_half_width=(0.0, 0.0, 0.0),
        # )

        domain_randomization_configuration = DomainRandomizationConfiguration(
            friction_ratio_range=(0.7, 1.3),
            friction_link_names=("ankle_left", "ankle_right"),
            friction_shared_across_links=True,
            default_joint_position_offset_range=(-0.01, 0.01),
            center_of_mass=CenterOfMassRandomization(
                link_names=("trunk_base", "jaw_soft"),
                x_range=(-0.003, 0.003),
                y_range=(-0.003, 0.003),
                z_range=(-0.003, 0.003),
            ),
            mass_inertia=MassInertiaRandomization(link_names=("trunk_base",), scale_range=(0.95, 1.05)),
            armature_scale_range=(0.9, 1.1),
        )
        # imu_link_name = "trunk_base"

        task = BeyondMimicTask(
            xml_path=mjcf_path,
            motion_library=motion_library,
            show_reference_motion=self.show_reference_motion,
            reference_motion_offset=self.reference_motion_offset,
            anchor_link_name="trunk_base",
            key_link_names=key_link_names,
            end_effector_link_names=("ankle_left", "ankle_right"),
            tracking_position_tolerance=0.045,
            tracking_linear_velocity_tolerance=0.15,
            tracking_height_threshold=0.0375,
            reset_perturbation_configuration=reset_perturbation_configuration if self.enable_randomization else None,
            sensor_configuration=sensor_configuration if self.enable_randomization else None,
        )
        return RobotEnvironment(
            engine,
            xml_path=mjcf_path,
            control_mode=control_mode,
            task=task,
            control_frequency=self.control_frequency,
            initial_pose=initial_pose,
            domain_randomization_configuration=domain_randomization_configuration if self.enable_randomization else None,
            push_configuration=push_configuration if self.enable_randomization else None,
            command_delay_configuration=command_delay_configuration,
        )
