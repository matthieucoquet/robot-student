from abc import ABC, abstractmethod
from dataclasses import dataclass

from robot_student.engine.genesis_engine import GenesisEngine
from robot_student.environment.environment import Environment


@dataclass(frozen=True, kw_only=True, slots=True)
class EnvironmentFactory(ABC):
    environment_count: int
    headless: bool = True
    control_frequency: int = 30
    simulation_frequency: int = 120

    def __post_init__(self) -> None:
        if self.control_frequency <= 0:
            raise ValueError("control_frequency must be positive")
        if self.simulation_frequency <= 0:
            raise ValueError("simulation_frequency must be positive")
        if self.simulation_frequency % self.control_frequency != 0:
            raise ValueError("simulation_frequency must be an integer multiple of control_frequency")

    def create_engine(self, *, use_cuda: bool, seed: int) -> Environment:
        engine = GenesisEngine(
            cuda_backend=use_cuda,
            show_viewer=not self.headless,
            seed=seed,
            simulation_frequency=self.simulation_frequency,
        )
        return engine

    @abstractmethod
    def create_environment(self, engine: GenesisEngine) -> Environment:
        """Create an environment"""
