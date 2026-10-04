import genesis as gs
import torch
from genesis.utils.misc import qd_to_torch

from robot_student.engine.xl330_m6 import Xl330ActuatorParameters


class Bam:
    """BAM actuator with a shared battery supply per environment.

    input_voltage is the nominal supply voltage. input_voltage_drop_resistance is in ohms;
    both accept scalars or tensors shaped [num_envs, 1]. minimum_input_voltage bounds
    the voltage after the load-dependent drop and defaults to 0.0 V.
    max_current optionally limits motor current in amps through the firmware PWM command.
    PWM saturation can prevent reaching this limit at high speed. None disables the limiter.

    Construct after scene.build(); construction configures actuator DOF properties.
    """

    @torch.no_grad()
    def __init__(
        self,
        entity,
        controlled_dofs_indices,
        input_voltage: torch.Tensor,
        *,
        input_voltage_drop_resistance: float | torch.Tensor | None = None,
        minimum_input_voltage: float = 0.0,
        max_current: float | None = None,
    ):
        if minimum_input_voltage is None:
            raise TypeError("minimum_input_voltage must be a float, not None")

        self._entity = entity
        self._controlled_dofs_indices = controlled_dofs_indices
        self._input_voltage = input_voltage
        self._input_voltage_drop_resistance = input_voltage_drop_resistance
        self._minimum_input_voltage = minimum_input_voltage

        self._actuator_parameters = Xl330ActuatorParameters(max_current=max_current)

        solver = entity.solver
        if solver.n_envs > 0 and not solver._options.batch_dofs_info:
            raise ValueError("BAM requires RigidOptions(batch_dofs_info=True) for batched environments")

        dofs_velocity = entity.get_dofs_velocity(self._controlled_dofs_indices)
        self._global_dofs_indices = torch.as_tensor(self._controlled_dofs_indices, dtype=torch.long, device=dofs_velocity.device)
        self._global_dofs_indices = self._global_dofs_indices + entity.dof_start
        self._previous_duty_cycle = torch.zeros_like(dofs_velocity)
        parameters = self._actuator_parameters
        dof_properties = torch.empty(len(self._controlled_dofs_indices), dtype=dofs_velocity.dtype, device=dofs_velocity.device)
        entity.set_dofs_armature(dof_properties.fill_(parameters.armature), self._controlled_dofs_indices)
        entity.set_dofs_damping(dof_properties.fill_(parameters.friction_viscous), self._controlled_dofs_indices)
        dof_properties.zero_()
        entity.set_dofs_kp(dof_properties, self._controlled_dofs_indices)
        entity.set_dofs_kv(dof_properties, self._controlled_dofs_indices)
        entity.set_dofs_stiffness(dof_properties, self._controlled_dofs_indices)
        entity.set_dofs_frictionloss(dof_properties, self._controlled_dofs_indices)
        force_upper_bounds = torch.full_like(dof_properties, float("inf"))
        entity.set_dofs_force_range(-force_upper_bounds, force_upper_bounds, self._controlled_dofs_indices)

    @property
    def actuator_parameters(self) -> Xl330ActuatorParameters:
        return self._actuator_parameters

    def reset(self, envs_indices=None) -> None:
        """Clear PWM history when resetting all or selected environments."""
        if envs_indices is None:
            self._previous_duty_cycle.zero_()
        else:
            self._previous_duty_cycle[envs_indices] = 0.0

    def _compute_control(self, target, dofs_position, dofs_velocity, input_voltage):
        parameters = self._actuator_parameters
        duty_cycle = (target - dofs_position) * parameters.kp * parameters.error_gain
        if parameters.max_current is not None:
            powered = input_voltage > 0.0
            safe_voltage = torch.where(powered, input_voltage, 1.0)
            duty_center = parameters.kt * dofs_velocity / safe_voltage
            duty_span = parameters.R * parameters.max_current / safe_voltage
            duty_cycle = torch.clamp(duty_cycle, duty_center - duty_span, duty_center + duty_span)
            duty_cycle = torch.where(powered, duty_cycle, 0.0)
        duty_cycle = torch.clamp(duty_cycle, -parameters.max_pwm, parameters.max_pwm)
        self._previous_duty_cycle.copy_(duty_cycle)
        return input_voltage * duty_cycle

    def _compute_motor_torque(self, volts, dofs_velocity):
        torque = self._actuator_parameters.kt * volts / self._actuator_parameters.R
        torque -= (self._actuator_parameters.kt**2) * dofs_velocity / self._actuator_parameters.R
        return torque

    def _compute_external_torque(self) -> torch.Tensor:
        """Return gravity, Coriolis and constraint loads, excluding DOF friction.

        Uses the previous solve and returns [num_envs, num_controlled_dofs] in
        entity-local DOF order. Removing friction prevents the load-dependent
        friction budget from feeding back on itself.
        """
        solver = self._entity.solver
        bias_force = qd_to_torch(solver.dyn_state.dofs.qf_bias, transpose=True)
        constraint_force = qd_to_torch(solver.dyn_state.dofs.qf_constraint, transpose=True)
        global_dofs_indices = self._global_dofs_indices

        constraint_solver = solver.constraint_solver
        frictionloss = qd_to_torch(solver.dyn_info.dofs.frictionloss, transpose=True)
        has_frictionloss = frictionloss > gs.EPS
        friction_rank = torch.cumsum(has_frictionloss, dim=-1) - 1
        equality_count = qd_to_torch(constraint_solver.n_constraints_equality)
        friction_count = qd_to_torch(constraint_solver.n_constraints_frictionloss)
        friction_rows = equality_count[:, None] + friction_rank
        friction_rows = friction_rows[:, global_dofs_indices]
        valid_rows = has_frictionloss[..., global_dofs_indices] & (friction_rank[..., global_dofs_indices] >= 0)
        valid_rows = valid_rows & (friction_rank[..., global_dofs_indices] < friction_count[:, None])

        constraint_effort = qd_to_torch(constraint_solver.efc_force, transpose=True)
        if constraint_effort.shape[-1] == 0:
            friction_force = torch.zeros_like(bias_force[:, global_dofs_indices])
        else:
            valid_rows = valid_rows & (friction_rows >= 0) & (friction_rows < constraint_effort.shape[-1])
            friction_rows = friction_rows.clamp(min=0, max=constraint_effort.shape[-1] - 1)
            friction_force = torch.where(valid_rows, torch.gather(constraint_effort, 1, friction_rows), 0.0)

        return -bias_force[:, global_dofs_indices] + constraint_force[:, global_dofs_indices] - friction_force

    def _compute_friction(
        self,
        motor_torque: torch.Tensor,
        external_torque: torch.Tensor,
        stribeck_coeff: torch.Tensor,
    ) -> torch.Tensor:
        parameters = self._actuator_parameters
        frictionloss = torch.full_like(motor_torque, parameters.friction_base)
        frictionloss = frictionloss + stribeck_coeff * parameters.friction_stribeck

        gearbox_torque = torch.abs(external_torque * parameters.load_friction_external - motor_torque * parameters.load_friction_motor)
        frictionloss = frictionloss + gearbox_torque
        gearbox_torque_stribeck = torch.abs(
            external_torque * parameters.load_friction_external_stribeck - motor_torque * parameters.load_friction_motor_stribeck
        )
        frictionloss = frictionloss + stribeck_coeff * gearbox_torque_stribeck

        absolute_external_torque = external_torque.abs()
        absolute_motor_torque = motor_torque.abs()
        quadratic_term = torch.where(
            absolute_motor_torque > absolute_external_torque,
            parameters.load_friction_external_quad * absolute_external_torque.square(),
            parameters.load_friction_motor_quad * absolute_motor_torque.square(),
        )
        return frictionloss + stribeck_coeff * quadratic_term

    @torch.no_grad()
    def step(self, target: torch.Tensor) -> torch.Tensor:
        """Return motor torque and update solver friction for the next physics step.

        Call once per physics step, then apply the returned torque with
        entity.control_dofs_force and advance the scene. Batched environments
        require RigidOptions(batch_dofs_info=True).
        """
        entity = self._entity
        solver = entity.solver
        dofs_position = entity.get_dofs_position(self._controlled_dofs_indices)
        dofs_velocity = entity.get_dofs_velocity(self._controlled_dofs_indices)

        previous_motor_torque = qd_to_torch(solver.dyn_state.dofs.qf_applied, transpose=True)[:, self._global_dofs_indices]

        input_voltage = self._input_voltage
        if self._input_voltage_drop_resistance is not None:
            battery_current = (self._previous_duty_cycle * previous_motor_torque / self._actuator_parameters.kt).sum(dim=-1, keepdim=True)
            input_voltage = input_voltage - self._input_voltage_drop_resistance * battery_current.clamp_min(0.0)
            input_voltage = input_voltage.clamp_min(self._minimum_input_voltage)
            if solver.n_envs == 0:
                input_voltage = input_voltage.squeeze(0)

        control = self._compute_control(target, dofs_position, dofs_velocity, input_voltage)
        motor_torque = self._compute_motor_torque(control, dofs_velocity)

        stribeck = torch.exp(
            -torch.pow(torch.abs(dofs_velocity) / self._actuator_parameters.dtheta_stribeck, self._actuator_parameters.alpha)
        )

        external_torque = self._compute_external_torque()
        frictionloss = self._compute_friction(previous_motor_torque, external_torque, stribeck)

        entity.set_dofs_frictionloss(frictionloss if solver.n_envs > 0 else frictionloss[0], self._controlled_dofs_indices)
        return motor_torque
