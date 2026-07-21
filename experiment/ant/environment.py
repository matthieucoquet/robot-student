from pathlib import Path

import torch

from robot_student.engine.control_mode import PositionControlMode, PositionControlSettings
from robot_student.environment import CharacterEnvironment, RunInDirectionTask


def setup_environment(engine, device: torch.device, environment_count: int = 10):
    mjcf_path = Path("./experiment/ant/ant.xml")

    joints_name = [
        "hip_1",
        "ankle_1",
        "hip_2",
        "ankle_2",
        "hip_3",
        "ankle_3",
        "hip_4",
        "ankle_4",
    ]
    joints_settings = {}
    for joint in joints_name:
        joints_settings[joint] = PositionControlSettings(kp=1500.0, kd=50.0, force_range=(-2000.0, 2000.0))
    control_mode = PositionControlMode(joints=joints_settings, action_limit_scale=1.1)
    task = RunInDirectionTask(device=device)
    initial_pose = (
        0.0,
        0.0,
        0.55,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        -1.0,
        0.0,
        -1.0,
        0.0,
        1.0,
    )

    return CharacterEnvironment(
        engine,
        mjcf_path,
        environment_count=environment_count,
        control_mode=control_mode,
        task=task,
        device=device,
        initial_pose=initial_pose,
    )
