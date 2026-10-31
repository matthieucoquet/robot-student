from dataclasses import replace
from functools import partial

from torch.optim import Adam

from robot_student.algorithm import PPOConfiguration, PPOFactory
from robot_student.model import MLP, PolicyConfiguration, ValueFunctionConfiguration
from robot_student.model.action import ActionBoundEnforcement, PositionTargetMode
from robot_student.model.weight_initializer import OrthogonalInitializer


def get_ppo_factory(
    *,
    actor_observation_keys: tuple[str, ...],
    critic_observation_keys: tuple[str, ...],
    compile_models: bool = False,
):
    policy = PolicyConfiguration(
        body_factory=partial(
            MLP,
            hidden_layers=[512, 512],
            weight_initializer=OrthogonalInitializer(head_gain=0.01),
        ),
        observation_keys=actor_observation_keys,
        action_key="control",
        action_bound_enforcement=ActionBoundEnforcement.BOUND_LOSS,
        position_target_mode=PositionTargetMode.DEFAULT_POSE_OFFSET,
        standard_deviation=0.1,
        learn_standard_deviation=False,
    )

    value_function = ValueFunctionConfiguration(
        body_factory=partial(
            MLP,
            hidden_layers=[512, 512],
            weight_initializer=OrthogonalInitializer(head_gain=1.0),
        ),
        observation_keys=critic_observation_keys,
    )

    learning_rate = 1e-4
    policy_optimizer = partial(Adam, lr=learning_rate)
    value_optimizer = partial(Adam, lr=learning_rate)

    return PPOFactory(
        configuration=PPOConfiguration(
            policy=policy,
            value_function=value_function,
            policy_optimizer=policy_optimizer,
            value_function_optimizer=value_optimizer,
            compile_models=compile_models,
            policy_epoch_count=2,
            policy_batch_size=2,
            value_epoch_count=2,
            value_batch_size=2,
            rollout_length=32,
            entropy_coefficient=0.0,
        )
    )


def get_beyond_mimic_ppo_factory(*, compile_models: bool = False) -> PPOFactory:
    configuration = get_ppo_factory(
        actor_observation_keys=(
            "motion_command",
            "measured_anchor_position_error",
            "measured_anchor_rotation_error",
            "measured_root_linear_velocity",
            "measured_imu_angular_velocity",
            "measured_projected_gravity",
            "measured_servo_encoder_position_offsets",
            "measured_servo_encoder_velocities",
            "previous_action",
        ),
        critic_observation_keys=(
            "motion_command",
            "simulated_anchor_position_error",
            "simulated_anchor_rotation_error",
            "simulated_root_linear_velocity",
            "simulated_imu_angular_velocity",
            "simulated_projected_gravity",
            "simulated_servo_encoder_position_offsets",
            "simulated_servo_encoder_velocities",
            "simulated_link_positions",
            "simulated_link_rotations",
            "previous_action",
        ),
        compile_models=compile_models,
    ).configuration
    # Use G1's BeyondMimic action parameterization with Microduck's BAM effort scale and smaller MLPs.
    return PPOFactory(
        configuration=replace(
            configuration,
            policy=replace(
                configuration.policy,
                action_bound_enforcement=ActionBoundEnforcement.NONE,
                position_target_mode=PositionTargetMode.EFFORT_SCALED_ACTION,
                standard_deviation=1.0,
                learn_standard_deviation=True,
            ),
            policy_optimizer=partial(Adam, lr=3e-5),
            value_function_optimizer=partial(Adam, lr=3e-5),
            entropy_coefficient=0.005,
        )
    )
