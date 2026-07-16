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
        joints_settings[joint] = PositionControlSettings(kp=1.0, kd=0.1, force_range=(-5.0, 5.0))
    control_mode = PositionControlMode(joints=joints_settings)
    task = RunInDirectionTask(device=device)

    return CharacterEnvironment(
        engine,
        mjcf_path,
        environment_count=environment_count,
        control_mode=control_mode,
        task=task,
        device=device,
    )
