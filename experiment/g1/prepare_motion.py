import logging
from pathlib import Path

from robot_student.run.motion_preprocess import MotionPreprocess

if __name__ == "__main__":
    experiment_path = Path(__file__).parent
    motion = MotionPreprocess(
        robot_path=experiment_path / "environment" / "mjcf" / "g1.xml",
        motion_folder=experiment_path / "dataset" / "motion_decode" / "test",
        output_folder=experiment_path / "dataset" / "preprocessed",
        debug_level=logging.INFO,
    )
    motion.run()
