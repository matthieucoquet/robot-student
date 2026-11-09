import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from robot_student.export import create_tracking_model, export_onnx
from robot_student.model import Policy, PolicyConfiguration
from robot_student.util.logging import configure_logging
from robot_student.util.storage import CheckpointStorage, RunContext, managed_storage

from .environment_factory import EnvironmentFactory
from .run_directory import find_run_directory


@dataclass(kw_only=True)
class OnnxExport:
    experiment_name: str
    run_name: str
    run_id: str
    environment_factory: EnvironmentFactory
    policy_configuration: PolicyConfiguration
    run_storage: CheckpointStorage
    output_path: Path | None = None
    checkpoint_iteration: int = -1
    motion_index: int = 0
    debug_level: int = logging.DEBUG

    @torch.no_grad()
    def run(self) -> None:
        configure_logging(self.debug_level)

        engine = self.environment_factory.create_engine(use_cuda=False, seed=0)
        environment = self.environment_factory.create_environment(engine=engine)

        policy = Policy(
            environment.schema,
            configuration=self.policy_configuration,
            device=environment.device,
        )
        model = create_tracking_model(policy, environment, motion_index=self.motion_index)
        run_directory = find_run_directory(experiment_name=self.experiment_name, run_name=self.run_name, run_id=self.run_id)
        context = RunContext(
            experiment_name=self.experiment_name,
            run_name=self.run_name,
            run_id=self.run_id,
            run_directory=run_directory,
            device=environment.device,
            is_evaluation=True,
        )
        output_path = self.output_path if self.output_path is not None else run_directory / "policy.onnx"

        with managed_storage(self.run_storage):
            self.run_storage.initialize(context)
            checkpoint = self.run_storage.load(iteration=self.checkpoint_iteration, device=environment.device)
            policy.load_state_dict(checkpoint["policy"])

            model.eval()
            export_onnx(model, environment.schema, output_path)
            logging.getLogger(__name__).info("Exported %s", output_path)
