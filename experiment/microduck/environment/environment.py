from dataclasses import dataclass
from pathlib import Path

from robot_student.engine.genesis_engine import GenesisEngine
from robot_student.environment import RobotEnvironment, RunInDirectionTask
from robot_student.environment.environment import Environment

from robot_student.run.environment_factory import EnvironmentFactory

from .robot_configuration import microduck_configuration


@dataclass(frozen=True, kw_only=True, slots=True)
class PPOEnvironmentFactory(EnvironmentFactory):
    def create_environment(
        self,
        engine: GenesisEngine,
    ) -> Environment:
        mjcf_path, control_mode, initial_pose, initial_joint_positions = microduck_configuration()

        task = RunInDirectionTask(
            device=engine.device,
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
        )
