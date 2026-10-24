from collections.abc import Sequence

import genesis as gs
import torch
from genesis.engine.entities import RigidEntity


class ServoJoints:
    """Clean motor and encoder readings in the supplied servo order.

    Each servo is a single-DOF joint, optionally paired with a same-axis
    ``passive_<name>_backlash`` hinge. Encoder readings sum the two coordinates.
    Joint indices are resolved once; reads are batched and retain no history.
    Noise, calibration bias, and observation delays belong to Sensor.
    """

    def __init__(self, entity: RigidEntity, joint_names: Sequence[str]) -> None:
        self._entity = entity
        self.joint_names = tuple(joint_names)
        joints_by_name = {joint.name: joint for joint in entity.joints}
        motor_dof_indices = []
        backlash_dof_indices = []
        backlash_servo_indices = []
        for servo_index, name in enumerate(self.joint_names):
            joint = joints_by_name[name]
            if joint.n_dofs != 1:
                raise ValueError(f"Servo joint {name!r} must have exactly one DOF")
            motor_dof_indices.append(joint.dofs_idx_local[0])
            backlash_joint = joints_by_name.get(f"passive_{name}_backlash")
            if backlash_joint is not None:
                if backlash_joint.n_dofs != 1 or backlash_joint.desc.dofs_act_gain.any():
                    raise ValueError(f"Backlash joint {backlash_joint.name!r} must have one unactuated DOF")
                backlash_dof_indices.append(backlash_joint.dofs_idx_local[0])
                backlash_servo_indices.append(servo_index)

        self.motor_dof_indices = torch.tensor(motor_dof_indices, dtype=torch.long, device=gs.device)
        self._backlash_dof_indices = torch.tensor(backlash_dof_indices, dtype=torch.long, device=gs.device)
        self._backlash_servo_indices = torch.tensor(backlash_servo_indices, dtype=torch.long, device=gs.device)

    @torch.no_grad()
    def get_motor_positions(self, environment_indices: torch.Tensor | None = None) -> torch.Tensor:
        return self._entity.get_dofs_position(self.motor_dof_indices, envs_idx=environment_indices)

    @torch.no_grad()
    def get_motor_velocities(self, environment_indices: torch.Tensor | None = None) -> torch.Tensor:
        return self._entity.get_dofs_velocity(self.motor_dof_indices, envs_idx=environment_indices)

    @torch.no_grad()
    def get_encoder_positions(self, environment_indices: torch.Tensor | None = None) -> torch.Tensor:
        positions = self.get_motor_positions(environment_indices)
        if self._backlash_dof_indices.numel():
            backlash = self._entity.get_dofs_position(self._backlash_dof_indices, envs_idx=environment_indices)
            positions.index_add_(-1, self._backlash_servo_indices, backlash)
        return positions

    @torch.no_grad()
    def get_encoder_velocities(self, environment_indices: torch.Tensor | None = None) -> torch.Tensor:
        velocities = self.get_motor_velocities(environment_indices)
        if self._backlash_dof_indices.numel():
            backlash = self._entity.get_dofs_velocity(self._backlash_dof_indices, envs_idx=environment_indices)
            velocities.index_add_(-1, self._backlash_servo_indices, backlash)
        return velocities
