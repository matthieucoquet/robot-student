import logging
from dataclasses import dataclass
from typing import Self

from robot_student.run.environment_factory import EnvironmentFactory
from robot_student.util.logging import configure_logging
from robot_student.util.seed import set_seed


@dataclass(frozen=True, kw_only=True, slots=True)
class RunConfiguration:
    experiment_name: str
    run_name: str
    seed: int
    use_cuda: bool
    iteration_count: int  # TODO in learner?
    checkpoint_interval: int   # TODO in learner?
    debug_level: int = logging.DEBUG
    environment: EnvironmentFactory
    learner: LearnerConfiguration

class Run:
    def __init__(self, configuration: RunConfiguration):
        configure_logging(configuration.debug_level)
        set_seed(configuration.seed)

        self.environment = configuration.environment.create(self.use_cuda, self.seed)


        self._build_learner(configuration.learner)
        self._




        pass
    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        # exception_type: type[BaseException] | None,
        # exception: BaseException | None,
        # traceback: TracebackType | None,
    ) -> None:
        # if exception_type is None:
        #     exit_code = 0
        # elif issubclass(exception_type, KeyboardInterrupt):
        #     exit_code = 130
        # else:
        #     exit_code = 1
        # self.close(exit_code=exit_code)
        pass
