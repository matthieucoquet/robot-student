from dataclasses import dataclass

from robot_student.engine.genesis_engine import GenesisEngine
from robot_student.environment import CharacterEnvironment, RunInDirectionTask
from robot_student.environment.environment import Environment
from robot_student.run.environment_factory import EnvironmentFactory

from .robot import g1_configuration


@dataclass(frozen=True, kw_only=True, slots=True)
class G1EnvironmentFactory(EnvironmentFactory):
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
