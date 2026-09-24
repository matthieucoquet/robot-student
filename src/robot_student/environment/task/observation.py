import torch
from genesis.utils.geom import transform_by_quat, transform_quat_by_quat

from robot_student.engine.robot_state import RobotState
from robot_student.environment.schema import TensorSchema
from robot_student.util.geometry import inverse_heading_rotation, quat_to_rot6d


def proprioception_schema(joint_count: int, key_link_count: int) -> TensorSchema:
    return TensorSchema(shape=(13 + 2 * joint_count + 3 * key_link_count,), data_type=torch.float32)


def proprioception_observation(state: RobotState, *, key_link_indices: torch.Tensor, global_observation: bool) -> torch.Tensor:
    root_position = state.root_position
    root_rotation = state.root_rotation
    root_velocity = state.root_velocity
    root_angular_velocity = state.root_angular_velocity

    key_link_positions = state.world_link_positions.index_select(-2, key_link_indices)
    relative_key_link_positions = key_link_positions - root_position.unsqueeze(-2)

    root_height = root_position[..., 2:3]
    if global_observation:
        root_rotation = quat_to_rot6d(root_rotation)
    else:
        inverse_heading = inverse_heading_rotation(root_rotation)
        relative_key_link_positions = transform_by_quat(relative_key_link_positions, inverse_heading.unsqueeze(-2))
        local_root_rotation = transform_quat_by_quat(root_rotation, inverse_heading)
        root_rotation = quat_to_rot6d(local_root_rotation)
        root_velocity = transform_by_quat(root_velocity, inverse_heading)
        root_angular_velocity = transform_by_quat(root_angular_velocity, inverse_heading)

    proprioception_components = [
        root_height,
        root_rotation,
        root_velocity,
        root_angular_velocity,
        state.joint_dof_positions,  # TODO: mimickit use 6D for each joint, relative to the rest/initial pose
        state.joint_dof_velocities,
        relative_key_link_positions.flatten(start_dim=-2),
    ]
    proprioception = torch.cat(proprioception_components, dim=-1)
    return proprioception
