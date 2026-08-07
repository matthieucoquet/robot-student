from functools import partial

from torch.optim import Adam

from robot_student.algorithm import PPOConfiguration, PPOFactory
from robot_student.model import MLP, PolicyConfiguration, ValueFunctionConfiguration
from robot_student.model.action import ActionBoundEnforcement
from robot_student.model.weight_initializer import OrthogonalInitializer


def get_ppo_factory():
    policy = PolicyConfiguration(
        body_factory=partial(
            MLP,
            hidden_layers=[256, 256],
            weight_initializer=OrthogonalInitializer(head_gain=0.01),
        ),
        observation_key="proprioception",
        action_key="control",
        action_bound_enforcement=ActionBoundEnforcement.BOUND_LOSS,
    )

    value_function = ValueFunctionConfiguration(
        body_factory=partial(
            MLP,
            hidden_layers=[256, 256],
            weight_initializer=OrthogonalInitializer(head_gain=1.0),
        ),
        observation_key="proprioception",
    )

    learning_rate = 3e-4
    policy_optimizer = partial(Adam, lr=learning_rate)
    value_optimizer = partial(Adam, lr=learning_rate)

    return PPOFactory(
        configuration=PPOConfiguration(
            policy=policy,
            value_function=value_function,
            policy_optimizer=policy_optimizer,
            value_function_optimizer=value_optimizer,
            rollout_length=32,
        )
    )
