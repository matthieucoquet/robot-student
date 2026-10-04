from pathlib import Path

from robot_student.engine.control_mode import BamControlMode


def microduck_configuration():
    mjcf_path = Path(__file__).parent / "mjcf" / "microduck" / "robot_walk.xml"

    control_mode = BamControlMode(
        input_voltage=7.4,
        input_voltage_drop_resistance=0.1,
        minimum_input_voltage=6.0,
        max_current=None,
        action_limit_scale=1.4,
    )

    initial_joint_positions = (
        0,
        -0.08726646259971647,
        -0.457924,
        -0.004940,
        0.452984,
        0.3490658503988659,
        0.3490658503988659,
        0,
        0,
        0,
        0.08726646259971647,
        0.457924,
        0.004940,
        -0.452984,
    )

    initial_pose = (
        0,
        0,
        0.12,
        1,
        0,
        0,
        0,
        *initial_joint_positions,
    )

    return mjcf_path, control_mode, initial_pose, initial_joint_positions
