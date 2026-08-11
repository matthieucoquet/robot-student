import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from robot_student.engine.genesis_engine import GenesisEngine
from robot_student.motion.motion import Motion
from robot_student.util.logging import configure_logging

ROOT_POSITION_COLUMNS = ("root_pos_x(m)", "root_pos_y(m)", "root_pos_z(m)")
ROOT_ROTATION_COLUMNS = ("root_rot_w", "root_rot_x", "root_rot_y", "root_rot_z")


@dataclass(kw_only=True)
class MotionPreprocess:
    robot_path: Path
    motion_folder: Path
    debug_level: int = logging.DEBUG
    use_cuda: bool = False
    seed: int = 0
    simulation_frequency: int = 120

    def _setup_scene(self):
        configure_logging(self.debug_level)

        self._engine = GenesisEngine(
            cuda_backend=self.use_cuda,
            show_viewer=not self.headless,
            seed=self.seed,
            simulation_frequency=self.simulation_frequency,
        )
        self._engine.add_ground_plane()
        self._character = self._engine.add_kinematic_character(self.robot_path)
        self._engine.build_scene(environment_count=1, env_spacing=(1.0, 1.0))
        self._motions = []

    def run(self):
        self._setup_scene()
        self._read_all_motions()

        for motion in self._motions:
            for i in range(motion.frame_count):
                # Process each frame of the motion
                pass

    def _read_all_motions(self):
        for file_path in sorted(self.motion_folder.glob("*.csv")):
            if file_path.is_file():
                motion = self._read_csv(file_path)
                self._motions.append(motion)

    @staticmethod
    def _read_csv(file_path: Path) -> Motion:
        with open(file_path, newline="", encoding="utf-8") as file:
            reader = csv.reader(file)
            try:
                column_names = next(reader)
            except StopIteration:
                raise ValueError(f"CSV file has no header: {file_path}") from None

            columns = [[] for _ in column_names]
            for row in reader:
                for column, value in zip(columns, row, strict=True):
                    column.append(float(value))

            column_tensors = {
                column_name: torch.tensor(values, dtype=torch.float32) for column_name, values in zip(column_names, columns, strict=True)
            }

        joint_dof_columns = tuple(column_name for column_name in column_tensors if column_name.startswith("dof_"))
        if not joint_dof_columns:
            raise ValueError(f"CSV file has no joint DOF columns: {file_path}")

        data = {
            "frequency": 120,
            "root_position": torch.stack([column_tensors[column_name] for column_name in ROOT_POSITION_COLUMNS], dim=-1),
            "root_rotation": torch.stack([column_tensors[column_name] for column_name in ROOT_ROTATION_COLUMNS], dim=-1),
            "joint_dof": torch.stack([column_tensors[column_name] for column_name in joint_dof_columns], dim=-1),
        }
        batch_size = data["root_position"].shape[:-1]
        return Motion(**data, batch_size=batch_size)
