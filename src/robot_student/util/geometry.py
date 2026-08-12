import torch
from genesis.utils.geom import _tc_transform_by_quat, axis_angle_to_quat


def quat_to_rot6d(rotation: torch.Tensor) -> torch.Tensor:
    """Convert ``(..., 4)`` wxyz quaternions to flattened first two rotation-matrix rows."""
    scale = 2 / rotation.square().sum(dim=-1, keepdim=True)
    scaled_vector = scale * rotation[..., 1:]

    w, x, y, z = rotation.unbind(dim=-1)
    scaled_x, scaled_y, scaled_z = scaled_vector.unbind(dim=-1)

    wx, wy, wz = w * scaled_x, w * scaled_y, w * scaled_z
    xx, xy, xz = x * scaled_x, x * scaled_y, x * scaled_z
    yy, yz = y * scaled_y, y * scaled_z
    zz = z * scaled_z

    return torch.stack(
        (
            1 - (yy + zz),
            xy - wz,
            xz + wy,
            xy + wz,
            1 - (xx + zz),
            yz - wx,
        ),
        dim=-1,
    )


def quat_to_rotation_vector(rotation: torch.Tensor) -> torch.Tensor:
    scalar = rotation[..., :1]
    vector = rotation[..., 1:]
    vector_norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    angle = 2 * torch.atan2(vector_norm, scalar.abs())

    epsilon = torch.finfo(rotation.dtype).eps
    scale = torch.where(vector_norm > epsilon, angle / vector_norm.clamp_min(epsilon), 2.0)
    shortest_path_sign = torch.where(scalar < 0, -1.0, 1.0)
    return shortest_path_sign * scale * vector


def heading_angle(rotation: torch.Tensor) -> torch.Tensor:
    forward = torch.tensor((1, 0, 0), device=rotation.device, dtype=rotation.dtype)
    transformed_forward = _tc_transform_by_quat(forward, rotation)

    heading = torch.atan2(transformed_forward[..., 1], transformed_forward[..., 0])
    return heading


def inverse_heading_rotation(root_rotation: torch.Tensor) -> torch.Tensor:
    angle = heading_angle(root_rotation)
    up = torch.tensor((0, 0, 1), device=root_rotation.device, dtype=root_rotation.dtype)

    heading_q = axis_angle_to_quat(-angle, up)
    return heading_q
