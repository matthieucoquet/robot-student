import logging

from robot_student.run.motion_preprocess import MotionPreprocess

if __name__ == "__main__":
    motion = MotionPreprocess(
        debug_level=logging.INFO,
    )
    motion.run()
