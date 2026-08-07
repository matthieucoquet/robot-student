from dataclasses import dataclass
from pathlib import Path

from robot_student.engine.control_mode import PositionControlMode, PositionControlSettings
from robot_student.engine.genesis_engine import GenesisEngine
from robot_student.environment import CharacterEnvironment, RunInDirectionTask
from robot_student.environment.environment import Environment
from robot_student.run.environment_factory import EnvironmentFactory


@dataclass(frozen=True, kw_only=True, slots=True)
class AntEnvironmentFactory(EnvironmentFactory):
    action_limit_scale: float = 1.1

    def create_environment(
        self,
        engine: GenesisEngine,
    ) -> Environment:
        mjcf_path = Path("./experiment/ant/mjcf/ant.xml")

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
            joints_settings[joint] = PositionControlSettings(kp=300.0, kd=10.0, armature=1.0, force_range=(-300.0, 300.0))
        control_mode = PositionControlMode(joints=joints_settings, action_limit_scale=self.action_limit_scale)
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
        task = RunInDirectionTask(device=engine.device, default_joint_positions=initial_pose[7:])

        return CharacterEnvironment(
            engine,
            mjcf_path,
            environment_count=self.environment_count,
            control_mode=control_mode,
            task=task,
            control_frequency=self.control_frequency,
            initial_pose=initial_pose,
        )
