from torch import nn


class ActorCritic(nn.Module):
    def __init__(self, policy: nn.Module, value: nn.Module) -> None:
        super().__init__()

        self._policy = policy
        self._value = value
