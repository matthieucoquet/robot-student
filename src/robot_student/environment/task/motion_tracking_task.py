from pathlib import Path

import torch

from robot_student.engine.genesis_engine import GenesisEngine
from robot_student.engine.robot import Robot
from robot_student.environment.task.task import Task
from robot_student.motion import MotionLibrary, ReferenceRobot


class MotionTrackingTask(Task):
    def __init__(
        self,
        xml_path: Path,
        motion_library: MotionLibrary,
        show_reference_motion: bool = False,
        reference_motion_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        super().__init__()
        self._xml_path = xml_path
        self._motion_library = motion_library
        self._reference_motion_offset = reference_motion_offset
        self._show_reference_motion = show_reference_motion

    def setup_scene(self, engine: GenesisEngine) -> None:
        self._kinematic_robot = None
        if self._show_reference_motion:
            self._kinematic_robot = engine.add_kinematic_robot(
                self._xml_path,
                position_offset=(0.0, 0.0, 0.0),
                color=(0.15, 0.55, 1.0, 0.7),
                name="reference_robot",
            )

        self._reference_robot = ReferenceRobot(
            engine.environment_count,
            self._motion_library,
            engine.time_step,
            engine.device,
            kinematic_robot=self._kinematic_robot,
            display_offset=self._reference_motion_offset,
        )
        self._reference_state = self._motion_library.get_state(
            torch.zeros(engine.environment_count, dtype=torch.int64, device=engine.device),
            torch.zeros(engine.environment_count, dtype=torch.float32, device=engine.device),
        )
        self._motion_finished = torch.zeros(engine.environment_count, dtype=torch.bool, device=engine.device)

    def initialize(
        self,
        *,
        robot: Robot,
        key_link_indices: torch.Tensor,
        simulation_steps_per_control_step: int,
        global_observation: bool,
    ) -> None:
        self._robot = robot
        self._key_link_indices = key_link_indices
        self._simulation_steps_per_control_step = simulation_steps_per_control_step
        self._global_observation = global_observation

    def reset(self, environment_indices: torch.Tensor) -> None:
        reference_state = self._reference_robot.reset(environment_indices=environment_indices)
        self._robot.set_state(reference_state, environment_indices=environment_indices)
        self._reference_state.copy_environments_(environment_indices, reference_state)
        self._motion_finished.index_fill_(0, environment_indices, False)

    def step(self, is_control_step: bool) -> None:
        if self._show_reference_motion:
            self._reference_state, self._motion_finished = self._reference_robot.step(1)
        elif is_control_step:
            self._reference_state, self._motion_finished = self._reference_robot.step(self._simulation_steps_per_control_step)
