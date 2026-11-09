import logging
from pathlib import Path

from robot_student.run import OnnxExport
from robot_student.util import WeightsAndBiasesStorage

from .environment.environment import BeyondMimicEnvironmentFactory
from .learner import get_beyond_mimic_ppo_factory

if __name__ == "__main__":
    motion_path = Path(__file__).parent / "dataset" / "preprocessed" / "BM_Stand_Up_00007.pt"

    environment = BeyondMimicEnvironmentFactory(
        headless=True,
        environment_count=1,
        enable_randomization=False,
        motion_path=motion_path,
    )
    learner_configuration = get_beyond_mimic_ppo_factory().configuration

    onnx_export = OnnxExport(
        experiment_name="microduck_beyondmimic",
        run_name="export",
        run_id="a13abae87d52e062",
        environment_factory=environment,
        policy_configuration=learner_configuration.policy,
        run_storage=WeightsAndBiasesStorage(),
        checkpoint_iteration=7000,
        debug_level=logging.INFO,
    )
    onnx_export.run()
