import logging
from pathlib import Path

from robot_student.run import MotionClipPlayer

if __name__ == "__main__":
    experiment_path = Path(__file__).parent
    player = MotionClipPlayer(
        robot_path=experiment_path / "environment" / "mjcf" / "g1.xml",
        motion_path=experiment_path / "dataset" / "preprocessed" / "BG_Normal_Walking_00001.pt",
        use_cuda=False,
        debug_level=logging.INFO,
    )
    player.run()
