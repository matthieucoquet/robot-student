from robot_student.engine.base_robot import BaseRobot
from robot_student.motion.motion_library import MotionLibrary


class ReferenceRobot(BaseRobot):
    def __init__(self, motion_library: MotionLibrary) -> None:
        super().__init__(character)
        self._motion_library = motion_library
