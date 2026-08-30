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
        self._motions = []

    def run(self):
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self._setup_scene()
        self._read_all_motions()

        # TODO batch is with several frame of the motion at the same time
        for source_path, raw_motion in self._motions:
            generalized_states = raw_motion.compute_generalized_states()
            world_link_positions = torch.empty(
                (raw_motion.frame_count, self._kinematic_robot.n_links, 3),
                dtype=generalized_states.root_position.dtype,
                device=self._engine.device,
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
            motion_clip = MotionClip(frequency=raw_motion.frequency, frames=frames.cpu())
            output_path = self.output_folder / source_path.with_suffix(".pt").name
            torch.save(motion_clip, output_path)

    def _read_all_motions(self):
        for file_path in sorted(self.motion_folder.glob("*.csv")):
            if file_path.is_file():
                self._motions.append((file_path, self._read_csv(file_path)))

    @staticmethod
    def _read_csv(file_path: Path) -> _RawMotionClip:
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

        return _RawMotionClip(
            frequency=120,
            root_position=torch.stack([column_tensors[column_name] for column_name in ROOT_POSITION_COLUMNS], dim=-1),
            root_rotation=torch.stack([column_tensors[column_name] for column_name in ROOT_ROTATION_COLUMNS], dim=-1),
            joint_dof_positions=torch.stack([column_tensors[column_name] for column_name in joint_dof_columns], dim=-1),
        )
