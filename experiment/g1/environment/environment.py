from dataclasses import dataclass
from pathlib import Path

from robot_student.engine.genesis_engine import GenesisEngine
from robot_student.engine.robot import CenterOfMassRandomization, DomainRandomizationConfiguration
from robot_student.engine.robot_state import NoiseConfiguration, UniformNoise
from robot_student.environment import RobotEnvironment, RunInDirectionTask
from robot_student.environment.environment import Environment
from robot_student.environment.robot_environment import PushConfiguration
from robot_student.environment.task.beyond_mimic_task import BeyondMimicTask
from robot_student.environment.task.deep_mimic_task import DeepMimicTask
from robot_student.environment.task.motion_tracking_task import ResetPerturbationConfiguration
from robot_student.motion import MotionLibrary, ReferenceSampling
from robot_student.run.environment_factory import EnvironmentFactory

from .robot_configuration import g1_configuration


@dataclass(frozen=True, kw_only=True, slots=True)
class PPOEnvironmentFactory(EnvironmentFactory):
    is_29_dof: bool = True

    def create_environment(
        self,
        engine: GenesisEngine,
    ) -> Environment:
        mjcf_path, control_mode, initial_pose, initial_joint_positions = g1_configuration(self.is_29_dof)

        task = RunInDirectionTask(
            device=engine.device,
            default_joint_positions=initial_joint_positions,
            height_range=(0.5, 1.5),
            target_height=0.7,
            target_speed=1.1,
            target_speed_weight=1.5,
            target_height_weight=0.0,
            facing_direction_weight=0.0,
            control_cost_weight=0.1,
            pose_cost_weight=0.5,
        )

        return RobotEnvironment(
            engine,
            mjcf_path,
            control_mode=control_mode,
            task=task,
            control_frequency=self.control_frequency,
            initial_pose=initial_pose,
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class DeepMimicEnvironmentFactory(EnvironmentFactory):
    is_29_dof: bool = True
    reference_sampling: ReferenceSampling = ReferenceSampling.UNIFORM
    show_reference_motion: bool = False
    reference_motion_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    reset_perturbation_configuration: ResetPerturbationConfiguration | None = None

    def create_environment(
        self,
        engine: GenesisEngine,
    ) -> Environment:
        mjcf_path, control_mode, initial_pose, initial_joint_positions = g1_configuration(self.is_29_dof)

        experiment_path = Path(__file__).parent.parent
        motion_path = experiment_path / "dataset" / "preprocessed" / "v1" / "BG_Normal_Walking_00001.pt"
        if not motion_path.is_file():
            raise FileNotFoundError(f"Preprocessed motion not found: {motion_path}")

        motion_library = MotionLibrary([motion_path], device=engine.device, reference_sampling=self.reference_sampling)

        joint_reward_weight = (
            1.0,  # Left hip pitch.
            1.0,  # Left hip roll.
            1.0,  # Left hip yaw.
            0.6,  # Left knee.
            0.5,  # Left ankle pitch.
            0.5,  # Left ankle roll.
            1.0,  # Right hip pitch.
            1.0,  # Right hip roll.
            1.0,  # Right hip yaw.
            0.6,  # Right knee.
            0.5,  # Right ankle pitch.
            0.5,  # Right ankle roll.
            1.0,  # Waist yaw.
            1.0,  # Waist roll.
            1.0,  # Waist pitch.
            1.0,  # Left shoulder pitch.
            1.0,  # Left shoulder roll.
            1.0,  # Left shoulder yaw.
            0.6,  # Left elbow.
            0.5,  # Left wrist roll.
            0.5,  # Left wrist pitch.
            0.5,  # Left wrist yaw.
            1.0,  # Right shoulder pitch.
            1.0,  # Right shoulder roll.
            1.0,  # Right shoulder yaw.
            0.6,  # Right elbow.
            0.5,  # Right wrist roll.
            0.5,  # Right wrist pitch.
            0.5,  # Right wrist yaw.
        )

        key_link_names = (
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
            # "left_elbow_link",
            # "right_elbow_link",
        )

        task = DeepMimicTask(
            device=engine.device,
            xml_path=mjcf_path,
            motion_library=motion_library,
            target_steps=[1, 2, 3],
            joint_reward_weight=joint_reward_weight,
            show_reference_motion=self.show_reference_motion,
            reference_motion_offset=self.reference_motion_offset,
            reset_perturbation_configuration=self.reset_perturbation_configuration,
        )

        return RobotEnvironment(
            engine,
            xml_path=mjcf_path,
            control_mode=control_mode,
            task=task,
            control_frequency=self.control_frequency,
            initial_pose=initial_pose,
            key_link_names=key_link_names,
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class BeyondMimicEnvironmentFactory(EnvironmentFactory):
    is_29_dof: bool = True
    reference_sampling: ReferenceSampling = ReferenceSampling.ADAPTIVE
    show_reference_motion: bool = False
    reference_motion_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    control_frequency: int = 50
    simulation_frequency: int = 200

    def create_environment(
        self,
        engine: GenesisEngine,
    ) -> Environment:
        mjcf_path, control_mode, initial_pose, initial_joint_positions = g1_configuration(self.is_29_dof)
        control_mode.action_limit_scale = None

        experiment_path = Path(__file__).parent.parent
        motion_path = experiment_path / "dataset" / "preprocessed" / "v1" / "BG_Normal_Walking_00001.pt"
        if not motion_path.is_file():
            raise FileNotFoundError(f"Preprocessed motion not found: {motion_path}")

        motion_library = MotionLibrary([motion_path], device=engine.device, reference_sampling=self.reference_sampling)

        anchor_link_name = "torso_link"
        key_link_names = (
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        )

        robot_state_noise = NoiseConfiguration(
            root_velocity=UniformNoise(half_width=0.5),
            root_angular_velocity=UniformNoise(half_width=0.2),
            joint_dof_positions=UniformNoise(half_width=0.01),
            joint_dof_velocities=UniformNoise(half_width=0.5),
            world_link_positions=UniformNoise(half_width=0.25),
            world_link_rotations=UniformNoise(half_width=0.05),
        )

        reset_perturbation_configuration = ResetPerturbationConfiguration(
            root_position_half_width=(0.05, 0.05, 0.01),
            root_rotation_half_width=(0.1, 0.1, 0.2),
            root_linear_velocity_half_width=(0.5, 0.5, 0.2),
            root_angular_velocity_half_width=(0.52, 0.52, 0.78),
            joint_position_half_width=0.1,
        )

        push_configuration = PushConfiguration(
            interval_seconds=2.0, linear_velocity_half_width=(0.5, 0.5, 0.2), angular_velocity_half_width=(0.52, 0.52, 0.78)
        )

        domain_randomization_configuration = DomainRandomizationConfiguration(
            friction_ratio_range=(0.3, 1.6),
            default_joint_position_offset_range=(-0.01, 0.01),
            center_of_mass=CenterOfMassRandomization(
                link_name="torso_link",
                x_range=(-0.025, 0.025),
                y_range=(-0.05, 0.05),
                z_range=(-0.05, 0.05),
            ),
        )

        task = BeyondMimicTask(
            xml_path=mjcf_path,
            motion_library=motion_library,
            show_reference_motion=self.show_reference_motion,
            reference_motion_offset=self.reference_motion_offset,
            reset_perturbation_configuration=reset_perturbation_configuration,
            anchor_link_name=anchor_link_name,
        )

        return RobotEnvironment(
            engine,
            xml_path=mjcf_path,
            control_mode=control_mode,
            task=task,
            control_frequency=self.control_frequency,
            initial_pose=initial_pose,
            key_link_names=key_link_names,
            noise_configuration=robot_state_noise,
            domain_randomization_configuration=domain_randomization_configuration,
            push_configuration=push_configuration,
        )
