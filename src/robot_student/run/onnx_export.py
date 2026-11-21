import logging
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from robot_student.environment.environment import Environment
from robot_student.environment.robot_environment import RobotEnvironment
from robot_student.environment.task.beyond_mimic_task import BeyondMimicTask
from robot_student.export import BeyondMimicPolicy
from robot_student.export.policy_contract import CONTRACT_METADATA_KEY, tracking_contract_json
from robot_student.model import Policy, PolicyConfiguration
from robot_student.util.logging import configure_logging
from robot_student.util.storage import CheckpointStorage, RunContext, managed_storage

from .environment_factory import EnvironmentFactory
from .run_directory import find_run_directory


@torch.no_grad()
def export_onnx(policy: Policy, environment: Environment, output_path: Path, *, motion_index: int = 0) -> None:
    """Export the policy with clip metadata embedded in ONNX and written beside it as JSON."""
    if not isinstance(environment, RobotEnvironment) or not isinstance(environment.task, BeyondMimicTask):
        raise ValueError("Motion tracking policies require a BeyondMimic environment")
    task = environment.task
    robot = environment.robot
    model = BeyondMimicPolicy(
        policy,
        task.motion_library,
        robot.servo_joints.motor_joint_dof_indices,
        anchor_link_index=robot.get_link_indices([task.anchor_link_name])[0],
        motion_index=motion_index,
    ).eval()

    schema = environment.schema
    batch_size = 1
    device = environment.device
    motion_time = torch.zeros(batch_size, dtype=torch.float32, device=device)
    initial_anchor_rotation = torch.tensor(
        (1.0, 0.0, 0.0, 0.0), dtype=schema.observations["anchor_rotation_error"].data_type, device=device
    ).expand(batch_size, -1)
    anchor_rotation = initial_anchor_rotation.clone()
    observations = {
        key: torch.zeros((batch_size, *schema.observations[key].shape), dtype=schema.observations[key].data_type, device=device)
        for key in policy.observation_keys
        if key not in model.computed_observation_keys
    }
    program = torch.onnx.export(
        model,
        (motion_time, initial_anchor_rotation, anchor_rotation),
        kwargs={"observations": observations},
        input_names=("motion_time", "initial_anchor_rotation", "anchor_rotation", *observations),
        output_names=(model.policy.action_key,),
        dynamo=True,
    )

    contract = tracking_contract_json(model, initial_previous_action=robot.default_control)
    program.model.metadata_props[CONTRACT_METADATA_KEY] = contract

    output_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path = output_path.with_suffix(".contract.json")
    with TemporaryDirectory(dir=output_path.parent) as directory:
        staged_model = Path(directory) / output_path.name
        staged_contract = Path(directory) / contract_path.name
        staged_model.write_bytes(program.model_proto.SerializeToString())
        staged_contract.write_text(contract, encoding="utf-8")
        staged_model.replace(output_path)
        staged_contract.replace(contract_path)


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

            export_onnx(policy, environment, output_path, motion_index=self.motion_index)
            logging.getLogger(__name__).info("Exported %s", output_path)
