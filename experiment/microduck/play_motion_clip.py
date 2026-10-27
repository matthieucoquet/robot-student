import logging
from pathlib import Path

from robot_student.run import MotionClipPlayer

if __name__ == "__main__":
    experiment_path = Path(__file__).resolve().parent

    camera_position = (0.45, -0.45, 0.3)

    player = MotionClipPlayer(
        robot_path=experiment_path / "environment" / "mjcf" / "microduck" / "robot_walk_backlash.xml",
        motion_path=experiment_path.parents[1] / "result" / "microduck_motion" / "preprocessed" / "BG_Normal_Walking_00001.pt",
        use_cuda=False,
        debug_level=logging.INFO,
        viewer_camera_position=camera_position,
        headless=True,
        recording_path=experiment_path.parents[1] / "result/microduck_motion/preview.mp4",
        recording_camera_position=camera_position,
    )
    player.run()
