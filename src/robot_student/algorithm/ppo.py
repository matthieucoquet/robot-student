import logging

import torch

from robot_student.model import ActorCritic


class PPO:
    def __init__(self, actor_critic: ActorCritic) -> None:
        self._actor_critic = actor_critic
        self._rollout_buffer = None
        self._logger = logging.getLogger(__name__)

    def train(self, iteration_count: int) -> None:
        self._actor_critic.train()

        for _ in range(iteration_count):
            self._collect_rollouts()
            self._logger.debug("PPO iteration")

            # TODO: if some iteration pass, save the model
            # do proper checkpointing to be able to resume training

    @torch.no_grad()
    def _collect_rollouts(self) -> None:
        # self._rollout_buffer.reset()
        pass
