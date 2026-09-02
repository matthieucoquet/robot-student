import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from robot_student.engine.genesis_engine import GenesisEngine
from robot_student.engine.kinematic_robot import GeneralizedRobotState, RobotState
from robot_student.motion.motion_clip import MotionClip
from robot_student.util.geometry import quat_angular_displacement
from robot_student.util.logging import configure_logging

ROOT_POSITION_COLUMNS = ("root_pos_x(m)", "root_pos_y(m)", "root_pos_z(m)")
ROOT_ROTATION_COLUMNS = ("root_rot_w", "root_rot_x", "root_rot_y", "root_rot_z")
DOF_COLUMN_PREFIX = "dof_"
DOF_COLUMN_SUFFIX = "(rad)"


@dataclass(frozen=True, kw_only=True, slots=True)
class _RawMotionClip:
    frequency: int
    root_position: torch.Tensor
    root_rotation: torch.Tensor
    joint_dof_positions: torch.Tensor

    @property
    def frame_count(self) -> int:
        return self.root_position.shape[0]

    @torch.no_grad()
    def compute_generalized_states(self) -> GeneralizedRobotState:
        if self.frame_count < 2:
            raise ValueError(f"At least two frames are required to compute velocities, got {self.frame_count}")

        root_velocity = torch.zeros_like(self.root_position)
        root_angular_velocity = torch.zeros_like(self.root_position)
        joint_dof_velocities = torch.zeros_like(self.joint_dof_positions)

        root_velocity[1:-1].copy_((self.root_position[2:] - self.root_position[:-2]) * (0.5 * self.frequency))
        root_velocity[0].copy_((self.root_position[1] - self.root_position[0]) * self.frequency)
        root_velocity[-1].copy_((self.root_position[-1] - self.root_position[-2]) * self.frequency)

        root_angular_velocity[1:-1].copy_(
            quat_angular_displacement(self.root_rotation[:-2], self.root_rotation[2:]) * (0.5 * self.frequency)
        )
        root_angular_velocity[0].copy_(quat_angular_displacement(self.root_rotation[0], self.root_rotation[1]) * self.frequency)
        root_angular_velocity[-1].copy_(quat_angular_displacement(self.root_rotation[-2], self.root_rotation[-1]) * self.frequency)

        joint_dof_velocities[1:-1].copy_((self.joint_dof_positions[2:] - self.joint_dof_positions[:-2]) * (0.5 * self.frequency))
        joint_dof_velocities[0].copy_((self.joint_dof_positions[1] - self.joint_dof_positions[0]) * self.frequency)
        joint_dof_velocities[-1].copy_((self.joint_dof_positions[-1] - self.joint_dof_positions[-2]) * self.frequency)

        return GeneralizedRobotState(
            root_position=self.root_position,
            root_rotation=self.root_rotation,
            joint_dof_positions=self.joint_dof_positions,
            root_velocity=root_velocity,
            root_angular_velocity=root_angular_velocity,
            joint_dof_velocities=joint_dof_velocities,
            batch_size=(self.frame_count,),
        )


@dataclass(kw_only=True)
class MotionPreprocess:
    robot_path: Path
    motion_folder: Path
    output_folder: Path
    motion_frequency: int = 120
    debug_level: int = logging.DEBUG
    headless: bool = True
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
        self._kinematic_robot = self._engine.add_kinematic_robot(self.robot_path)
        self._engine.build_scene(environment_count=1, env_spacing=(1.0, 1.0))

    @torch.inference_mode()
    def run(self) -> None:
        source_paths = sorted(path for path in self.motion_folder.rglob("*.csv") if path.is_file())
        if not source_paths:
            raise ValueError(f"No CSV motion files found under {self.motion_folder}")

        self.output_folder.mkdir(parents=True, exist_ok=True)
        self._setup_scene()
        logger = logging.getLogger(__name__)
        for source_path in source_paths:
            raw_motion = self._read_csv(source_path)
            motion_clip = self._preprocess_motion(raw_motion)
            relative_output_path = source_path.relative_to(self.motion_folder).with_suffix(".pt")
            output_path = self.output_folder / relative_output_path

            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(motion_clip, output_path)
            logger.info("Preprocessed %s -> %s", source_path, output_path)

    def _preprocess_motion(self, raw_motion: _RawMotionClip) -> MotionClip:
        generalized_states = raw_motion.compute_generalized_states()
        world_link_positions = torch.empty(
            (raw_motion.frame_count, self._kinematic_robot.n_links, 3),
            dtype=generalized_states.root_position.dtype,
            device="cpu",
        )

        for frame_index in range(raw_motion.frame_count):
            state = self._kinematic_robot.set_state(generalized_states[frame_index])
            world_link_positions[frame_index].copy_(state.world_link_positions[0])

        frames = RobotState(
            root_position=generalized_states.root_position,
            root_rotation=generalized_states.root_rotation,
            joint_dof_positions=generalized_states.joint_dof_positions,
            root_velocity=generalized_states.root_velocity,
            root_angular_velocity=generalized_states.root_angular_velocity,
            joint_dof_velocities=generalized_states.joint_dof_velocities,
            world_link_positions=world_link_positions,
            batch_size=generalized_states.batch_size,
        )
        return MotionClip(frequency=raw_motion.frequency, frames=frames.cpu())

    def _read_csv(self, file_path: Path) -> _RawMotionClip:
        with open(file_path, newline="", encoding="utf-8") as file:
            reader = csv.reader(file)
            try:
                column_names = [column_name.strip() for column_name in next(reader)]
            except StopIteration:
                raise ValueError(f"CSV file has no header: {file_path}") from None

            columns = [[] for _ in column_names]
            for row_number, row in enumerate(reader, start=2):
                if len(row) != len(column_names):
                    raise ValueError(f"CSV row {row_number} in {file_path} has {len(row)} values, expected {len(column_names)}")
                for column, value in zip(columns, row, strict=True):
                    try:
                        column.append(float(value))
                    except ValueError as error:
                        raise ValueError(f"CSV row {row_number} in {file_path} contains a non-numeric value: {value!r}") from error

        column_tensors = {
            column_name: torch.tensor(values, dtype=torch.float32) for column_name, values in zip(column_names, columns, strict=True)
        }
        missing_root_columns = set((*ROOT_POSITION_COLUMNS, *ROOT_ROTATION_COLUMNS)) - column_tensors.keys()
        if missing_root_columns:
            missing_columns_text = ", ".join(sorted(missing_root_columns))
            raise ValueError(f"CSV file {file_path} is missing root columns: {missing_columns_text}")

        joint_columns_by_name: dict[str, str] = {}
        for column_name in column_names:
            joint_dof_name = _parse_joint_dof_column_name(column_name)
            if joint_dof_name is None:
                continue
            joint_columns_by_name[joint_dof_name] = column_name

        expected_joint_dof_names = self._kinematic_robot.joint_dof_names
        missing_joint_dofs = set(expected_joint_dof_names) - joint_columns_by_name.keys()
        unknown_joint_dofs = joint_columns_by_name.keys() - set(expected_joint_dof_names)
        if missing_joint_dofs or unknown_joint_dofs:
            problem_parts = []
            if missing_joint_dofs:
                problem_parts.append(f"missing: {', '.join(sorted(missing_joint_dofs))}")
            if unknown_joint_dofs:
                problem_parts.append(f"unknown: {', '.join(sorted(unknown_joint_dofs))}")
            raise ValueError(f"CSV joint DOFs do not match the robot in {file_path} ({'; '.join(problem_parts)})")

        root_position = torch.stack([column_tensors[column_name] for column_name in ROOT_POSITION_COLUMNS], dim=-1)
        root_rotation = torch.stack([column_tensors[column_name] for column_name in ROOT_ROTATION_COLUMNS], dim=-1)
        joint_dof_positions = torch.stack(
            [column_tensors[joint_columns_by_name[joint_dof_name]] for joint_dof_name in expected_joint_dof_names], dim=-1
        )

        return _RawMotionClip(
            frequency=self.motion_frequency,
            root_position=root_position,
            root_rotation=root_rotation,
            joint_dof_positions=joint_dof_positions,
        )


def _parse_joint_dof_column_name(column_name: str) -> str | None:
    if not column_name.startswith(DOF_COLUMN_PREFIX):
        return None
    joint_dof_name = column_name.removeprefix(DOF_COLUMN_PREFIX)
    if joint_dof_name.endswith(DOF_COLUMN_SUFFIX):
        joint_dof_name = joint_dof_name.removesuffix(DOF_COLUMN_SUFFIX)
    if not joint_dof_name:
        raise ValueError(f"Invalid joint DOF column name: {column_name!r}")
    return joint_dof_name
