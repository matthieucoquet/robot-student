import logging

import torch

from robot_student.model import ActorCritic


class PPO:
    def __init__(self, actor_critic: ActorCritic) -> None:
        self._actor_critic = actor_critic
        self._rollout_buffer = None
        self._logger = logging.getLogger(__name__)

    def train(self, experiment_storage, iteration_count: int, checkpoint_interval: int) -> None:
        self._actor_critic.train()

        for i in range(iteration_count):
            self._collect_rollouts()
            self._logger.debug("PPO iteration")

            experiment_storage.metrics.log_scalar("test", i * 0.1, i)

            if i % checkpoint_interval == 0:
                experiment_storage.checkpoint.save({"actor_critic": self._actor_critic.state_dict()}, i)

    @torch.no_grad()
    def _collect_rollouts(self) -> None:
        # self._rollout_buffer.reset()
        pass
