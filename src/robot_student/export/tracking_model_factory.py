from robot_student.environment.environment import Environment
from robot_student.environment.robot_environment import RobotEnvironment
from robot_student.environment.task.beyond_mimic_task import BeyondMimicTask
from robot_student.model import Policy

from .onnx import OnnxTrackingModel


def create_tracking_model(
    policy: Policy,
    environment: Environment,
    *,
    motion_index: int = 0,
) -> OnnxTrackingModel:
    """Build a tracking model from a BeyondMimic environment's motion and servo mapping."""
    if not isinstance(environment, RobotEnvironment) or not isinstance(environment.task, BeyondMimicTask):
        raise ValueError("ONNX motion export requires a BeyondMimic environment")

    return OnnxTrackingModel(
        policy,
        environment.task.motion_library,
        environment.robot.servo_joints.motor_joint_dof_indices,
        motion_index=motion_index,
    )
