from dataclasses import dataclass
from pathlib import Path

from robot_student.engine.genesis_engine import GenesisEngine
from robot_student.environment import CharacterEnvironment, RunInDirectionTask
from robot_student.environment.environment import Environment
from robot_student.environment.motion_tracking_environment import DeepMimicTask, MotionTrackingEnvironment
from robot_student.motion import MotionLibrary
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

        return CharacterEnvironment(
            engine,
            mjcf_path,
            environment_count=self.environment_count,
            control_mode=control_mode,
            task=task,
            control_frequency=self.control_frequency,
            initial_pose=initial_pose,
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class TrackerEnvironmentFactory(EnvironmentFactory):
    is_29_dof: bool = True
    random_reference_sampling: bool = True

    def create_environment(
        self,
        engine: GenesisEngine,
    ) -> Environment:
        mjcf_path, control_mode, initial_pose, initial_joint_positions = g1_configuration(self.is_29_dof)

        experiment_path = Path(__file__).parent.parent
        motion_paths = [experiment_path / "dataset" / "preprocessed" / "v1" / "BG_Normal_Walking_00001.pt"]

        motion_library = MotionLibrary(motion_paths, device=engine.device)

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
        )

        task = DeepMimicTask(device=engine.device, joint_reward_weight=joint_reward_weight)

        return MotionTrackingEnvironment(
            engine,
            motion_library=motion_library,
            xml_path=mjcf_path,
            environment_count=self.environment_count,
            control_mode=control_mode,
            task=task,
            control_frequency=self.control_frequency,
            initial_pose=initial_pose,
            key_link_names=key_link_names,
            target_steps=[1, 2, 3],
            random_reference_sampling=self.random_reference_sampling,
        )
