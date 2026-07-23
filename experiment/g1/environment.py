from pathlib import Path

import torch

from robot_student.engine.control_mode import PositionControlMode, PositionControlSettings
from robot_student.environment import CharacterEnvironment, RunInDirectionTask


def setup_environment(engine, device: torch.device, environment_count: int = 10):
    mjcf_path = Path(__file__).parent / "mjcf" / "g1_23dof.xml"

    joint_settings = {}
    joints_name_group = [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
    ]
    for joint in joints_name_group:
        joint_settings[joint] = PositionControlSettings(
            kp=14.25062309787429,
            kd=0.907222843292423,
            force_range=(-25.0, 25.0),
        )

    joints_name_group = [
        "left_hip_pitch_joint",
        "left_hip_yaw_joint",
        "right_hip_pitch_joint",
        "right_hip_yaw_joint",
        "waist_yaw_joint",
    ]
    for joint in joints_name_group:
        joint_settings[joint] = PositionControlSettings(
            kp=14.25062309787429,
            kd=0.907222843292423,
            force_range=(-25.0, 25.0),
        )

    joints_name_group = [
        "left_hip_pitch_joint",
        "left_hip_yaw_joint",
        "right_hip_pitch_joint",
        "right_hip_yaw_joint",
        "waist_yaw_joint",
    ]
    for joint in joints_name_group:
        joint_settings[joint] = PositionControlSettings(
            kp=40.17923863450712,
            kd=2.557889775413375,
            force_range=(-88.0, 88.0),
        )

    joints_name_group = [
        "left_hip_roll_joint",
        "left_knee_joint",
        "right_hip_roll_joint",
        "right_knee_joint",
    ]
    for joint in joints_name_group:
        joint_settings[joint] = PositionControlSettings(
            kp=99.09842777666111,
            kd=6.308801853496639,
            force_range=(-139.0, 139.0),
        )

    joints_name_group = [
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    ]
    for joint in joints_name_group:
        joint_settings[joint] = PositionControlSettings(
            kp=28.50124619574858,
            kd=1.814445686584846,
            force_range=(-50.0, 50.0),
        )

    control_mode = PositionControlMode(joints=joint_settings, action_limit_scale=1.1)

    initial_pose = (
        0.0,  # Root x.
        0.0,  # Root y.
        0.8,  # Root z.
        1.0,  # Root quaternion w.
        0.0,  # Root quaternion x.
        0.0,  # Root quaternion y.
        0.0,  # Root quaternion z.
        -0.1,  # Left hip pitch.
        0.0,  # Left hip roll.
        0.0,  # Left hip yaw.
        0.3,  # Left knee.
        -0.2,  # Left ankle pitch.
        0.0,  # Left ankle roll.
        -0.1,  # Right hip pitch.
        0.0,  # Right hip roll.
        0.0,  # Right hip yaw.
        0.3,  # Right knee.
        -0.2,  # Right ankle pitch.
        0.0,  # Right ankle roll.
        0.0,  # Waist yaw.
        0.35,  # Left shoulder pitch.
        0.18,  # Left shoulder roll.
        0.0,  # Left shoulder yaw.
        0.87,  # Left elbow.
        0.0,  # Left wrist roll.
        0.35,  # Right shoulder pitch.
        -0.18,  # Right shoulder roll.
        0.0,  # Right shoulder yaw.
        0.87,  # Right elbow.
        0.0,  # Right wrist roll.
    )

    task = RunInDirectionTask(device=device)

    return CharacterEnvironment(
        engine,
        mjcf_path,
        environment_count=environment_count,
        control_mode=control_mode,
        task=task,
        device=device,
        initial_pose=initial_pose,
    )
