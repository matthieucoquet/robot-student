"""Retarget a MotionDecode G1 CSV with the optional GMR dependency."""

import csv
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

REPOSITORY_PATH = Path(__file__).resolve().parents[2]
MICRODUCK_PATH = Path(__file__).resolve().parent
ROOT_COLUMNS = ("root_pos_x(m)", "root_pos_y(m)", "root_pos_z(m)", "root_rot_w", "root_rot_x", "root_rot_y", "root_rot_z")
FOOT_NAMES = ("left_foot", "right_foot")


@dataclass(frozen=True, kw_only=True, slots=True)
class RetargetingConfiguration:
    motion_scale: float = 0.15
    stance_half_width: float = 0.042
    ground_percentile: float = 1.0
    ground_clearance_m: float = 0.0002
    root_position_weight: float = 100.0
    root_rotation_weight: float = 5.0
    foot_position_weight: float = 200.0
    foot_rotation_weight: float = 5.0
    head_joint_positions: dict[str, float] = field(
        default_factory=lambda: {
            "neck_pitch": 0.3490658503988659,
            "head_pitch": 0.3490658503988659,
            "head_yaw": 0.0,
            "head_roll": 0.0,
        }
    )
    leg_joint_positions: dict[str, float] = field(
        default_factory=lambda: {
            "left_hip_yaw": 0.0,
            "left_hip_roll": -0.08726646259971647,
            "left_hip_pitch": -0.457924,
            "left_knee": -0.00494,
            "left_ankle": 0.452984,
            "right_hip_yaw": 0.0,
            "right_hip_roll": 0.08726646259971647,
            "right_hip_pitch": 0.457924,
            "right_knee": 0.00494,
            "right_ankle": -0.452984,
        }
    )


def joint_names(model: mujoco.MjModel) -> list[str]:
    if model.njnt == 0 or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE or np.any(model.jnt_type[1:] != mujoco.mjtJoint.mjJNT_HINGE):
        raise ValueError("Expected one free root followed by named hinge joints")
    return [model.joint(index).name for index in range(1, model.njnt)]


def read_motion(path: Path, model: mujoco.MjModel) -> np.ndarray:
    expected_columns = [*ROOT_COLUMNS, *(f"dof_{name}(rad)" for name in joint_names(model))]
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        columns = [column.strip() for column in next(reader, [])]
        if len(set(columns)) != len(columns):
            raise ValueError("Motion CSV contains duplicate columns")
        if set(columns) != set(expected_columns):
            raise ValueError(f"Motion CSV columns do not match the source robot: {sorted(set(columns) ^ set(expected_columns))}")
        rows = np.asarray(list(reader), dtype=np.float64)
    if rows.ndim != 2 or len(rows) < 2 or rows.shape[1] != len(columns) or not np.isfinite(rows).all():
        raise ValueError("Motion must contain at least two finite frames with all expected columns")
    positions = rows[:, [columns.index(name) for name in expected_columns]]
    quaternion_lengths = np.linalg.norm(positions[:, 3:7], axis=1, keepdims=True)
    if np.any(quaternion_lengths < 1e-8):
        raise ValueError("Motion contains a zero root quaternion")
    positions[:, 3:7] /= quaternion_lengths
    return positions


def source_body_poses(model: mujoco.MjModel, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return root and sole poses, with the initial XY position and heading removed."""
    data = mujoco.MjData(model)
    body_positions = np.empty((len(positions), 3, 3))
    body_quaternions = np.empty((len(positions), 3, 4))
    foot_indices = [model.site(name).id for name in FOOT_NAMES]
    initial_heading = Rotation.from_quat(positions[0, 3:7], scalar_first=True).as_euler("xyz")[2]
    heading_inverse = Rotation.from_euler("z", -initial_heading)
    origin = np.array([*positions[0, :2], 0.0])
    for frame_index, frame in enumerate(positions):
        data.qpos[:] = frame
        mujoco.mj_kinematics(model, data)
        body_positions[frame_index] = heading_inverse.apply(np.vstack((frame[:3], data.site_xpos[foot_indices])) - origin)
        rotations = [Rotation.from_quat(frame[3:7], scalar_first=True)]
        rotations.extend(Rotation.from_matrix(data.site_xmat[index].reshape(3, 3)) for index in foot_indices)
        body_quaternions[frame_index] = [(heading_inverse * rotation).as_quat(scalar_first=True) for rotation in rotations]
    return body_positions, body_quaternions


def prepare_targets(
    source_model: mujoco.MjModel, positions: np.ndarray, configuration: RetargetingConfiguration
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return poses scaled to Microduck and the ground height in source meters."""
    body_positions, body_quaternions = source_body_poses(source_model, positions)
    source_ground = float(np.percentile(body_positions[:, 1:, 2], configuration.ground_percentile))
    body_positions[:, :, 2] -= source_ground
    body_positions *= configuration.motion_scale
    # Preserve the G1 foot trajectories while adjusting the resting stance to Microduck's hip width.
    neutral_data = mujoco.MjData(source_model)
    mujoco.mj_kinematics(source_model, neutral_data)
    heading = Rotation.from_quat(body_quaternions[:, 0], scalar_first=True).as_euler("xyz")[:, 2]
    lateral_direction = np.column_stack((-np.sin(heading), np.cos(heading), np.zeros(len(heading))))
    for foot_index, (foot_name, side) in enumerate(zip(FOOT_NAMES, (1.0, -1.0), strict=True), start=1):
        source_half_width = abs(neutral_data.site_xpos[source_model.site(foot_name).id, 1])
        offset = configuration.stance_half_width - configuration.motion_scale * source_half_width
        body_positions[:, foot_index] += side * offset * lateral_direction
    return body_positions, body_quaternions, source_ground


def gmr_configuration(model: mujoco.MjModel, configuration: RetargetingConfiguration) -> dict:
    matches = {"trunk_base": ["pelvis", configuration.root_position_weight, configuration.root_rotation_weight, [0, 0, 0], [1, 0, 0, 0]]}
    for foot_name in FOOT_NAMES:
        site = model.site(foot_name)
        body_name = model.body(site.bodyid[0]).name
        matches[body_name] = [
            foot_name,
            configuration.foot_position_weight,
            configuration.foot_rotation_weight,
            (-site.pos).tolist(),
            Rotation.from_quat(site.quat, scalar_first=True).inv().as_quat(scalar_first=True).tolist(),
        ]
    return {
        "human_height_assumption": 1.0,
        "human_root_name": "pelvis",
        "robot_root_name": "trunk_base",
        "human_scale_table": {name: 1.0 for name in ("pelvis", *FOOT_NAMES)},
        "ground_height": 0.0,
        "use_ik_match_table1": True,
        "use_ik_match_table2": False,
        "ik_match_table1": matches,
        "ik_match_table2": {},
    }


def retarget(
    robot_path: Path,
    config_path: Path,
    configuration: RetargetingConfiguration,
    body_positions: np.ndarray,
    body_quaternions: np.ndarray,
    frequency: int,
) -> np.ndarray:
    from general_motion_retargeting import GeneralMotionRetargeting
    from general_motion_retargeting.params import IK_CONFIG_DICT, ROBOT_XML_DICT

    # Register paths at runtime; the external checkout and its solver remain unmodified.
    ROBOT_XML_DICT["microduck"] = robot_path.resolve()
    IK_CONFIG_DICT["g1"] = {"microduck": config_path.resolve()}
    retargeter = GeneralMotionRetargeting(src_human="g1", tgt_robot="microduck", solver="daqp", verbose=False, use_velocity_limit=True)
    retargeter.model.opt.timestep = 1.0 / frequency
    # GMR's limit applies to each integration. One solve per frame preserves the 3*pi rad/s cap.
    retargeter.max_iter = 0
    initial_pose = retargeter.model.qpos0.copy()
    initial_pose[:3] = body_positions[0, 0]
    initial_pose[3:7] = body_quaternions[0, 0]
    for name, position in (configuration.leg_joint_positions | configuration.head_joint_positions).items():
        initial_pose[retargeter.model.joint(name).qposadr[0]] = position
    retargeter.configuration.update(initial_pose)
    output = np.empty((len(body_positions), retargeter.model.nq))
    for frame_index, (frame_positions, frame_quaternions) in enumerate(zip(body_positions, body_quaternions, strict=True)):
        targets = dict(zip(("pelvis", *FOOT_NAMES), zip(frame_positions, frame_quaternions, strict=True), strict=True))
        if frame_index == 0:
            for _ in range(120):
                retargeter.retarget(targets.copy())
        output[frame_index] = retargeter.retarget(targets)
        if (frame_index + 1) % 240 == 0:
            print(f"Retargeted {frame_index + 1}/{len(output)} frames", flush=True)
    # Head joints form a separate branch and are absent from the leg/root IK objectives.
    for name, position in configuration.head_joint_positions.items():
        output[:, retargeter.model.joint(name).qposadr[0]] = position
    return output


def write_motion(path: Path, model: mujoco.MjModel, positions: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ",".join([*ROOT_COLUMNS, *(f"dof_{name}(rad)" for name in joint_names(model))])
    np.savetxt(path, positions, delimiter=",", header=header, comments="", fmt="%.10g")


def with_zero_backlash(model: mujoco.MjModel, backlash_model: mujoco.MjModel, positions: np.ndarray) -> np.ndarray:
    output = np.zeros((len(positions), backlash_model.nq))
    output[:, :7] = positions[:, :7]
    for name in joint_names(backlash_model):
        if name.startswith("passive_") and name.endswith("_backlash"):
            continue
        output[:, backlash_model.joint(name).qposadr[0]] = positions[:, model.joint(name).qposadr[0]]
    return output


def sole_geometry(model: mujoco.MjModel) -> list[tuple[int, np.ndarray]]:
    geometry = []
    for name in ("left_foot_collision", "right_foot_collision"):
        geom = model.geom(name)
        mesh_index = geom.dataid[0]
        first_vertex = model.mesh_vertadr[mesh_index]
        vertices = model.mesh_vert[first_vertex : first_vertex + model.mesh_vertnum[mesh_index]]
        geometry.append((geom.id, vertices))
    return geometry


def sole_heights(data: mujoco.MjData, geometry: list[tuple[int, np.ndarray]]) -> np.ndarray:
    return np.array([np.min(vertices @ data.geom_xmat[index].reshape(3, 3)[2] + data.geom_xpos[index, 2]) for index, vertices in geometry])


def adapt_sole_clearance(model: mujoco.MjModel, body_positions: np.ndarray, body_quaternions: np.ndarray, clearance: float) -> float:
    """Raise foot targets in Microduck meters to clear the floor at each target orientation."""
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    maximum_lift = 0.0
    for foot_index, (foot_name, (geom_index, vertices)) in enumerate(zip(FOOT_NAMES, sole_geometry(model), strict=True), start=1):
        site_index = model.site(foot_name).id
        world_vertices = vertices @ data.geom_xmat[geom_index].reshape(3, 3).T + data.geom_xpos[geom_index]
        sole_vertices = (world_vertices - data.site_xpos[site_index]) @ data.site_xmat[site_index].reshape(3, 3)
        rotations = Rotation.from_quat(body_quaternions[:, foot_index], scalar_first=True).as_matrix()
        minimum_offsets = np.min(sole_vertices @ rotations[:, 2, :].T, axis=0)
        lift = np.maximum(0.0, clearance - minimum_offsets - body_positions[:, foot_index, 2])
        body_positions[:, foot_index, 2] += lift
        maximum_lift = max(maximum_lift, float(lift.max()))
    return maximum_lift


def align_to_ground(model: mujoco.MjModel, positions: np.ndarray, clearance: float) -> float:
    """Raise each root only as needed to keep the complete sole meshes above the floor."""
    data = mujoco.MjData(model)
    geometry = sole_geometry(model)
    maximum_lift = 0.0
    for frame in positions:
        data.qpos[:] = frame
        mujoco.mj_kinematics(model, data)
        lift = max(0.0, clearance - float(sole_heights(data, geometry).min()))
        frame[2] += lift
        maximum_lift = max(maximum_lift, lift)
    return maximum_lift


def motion_statistics(model: mujoco.MjModel, positions: np.ndarray, frequency: int) -> dict:
    data = mujoco.MjData(model)
    feet = np.empty((len(positions), 2, 3))
    heights = np.empty((len(positions), 2))
    geometry = sole_geometry(model)
    for frame_index, frame in enumerate(positions):
        data.qpos[:] = frame
        mujoco.mj_kinematics(model, data)
        feet[frame_index] = data.site_xpos[[model.site(name).id for name in FOOT_NAMES]]
        heights[frame_index] = sole_heights(data, geometry)
    lower = model.jnt_range[1:, 0]
    upper = model.jnt_range[1:, 1]
    return {
        "frame_count": len(positions),
        "frequency": frequency,
        "duration_seconds": (len(positions) - 1) / frequency,
        "minimum_foot_site_height_m": float(feet[:, :, 2].min()),
        "maximum_foot_site_height_m": float(feet[:, :, 2].max()),
        "minimum_sole_height_m": float(heights.min()),
        "root_height_range_m": [float(positions[:, 2].min()), float(positions[:, 2].max())],
        "root_travel_m": float(np.linalg.norm(positions[-1, :2] - positions[0, :2])),
        "maximum_joint_speed_rad_per_second": float(np.abs(np.diff(positions[:, 7:], axis=0)).max() * frequency),
        "maximum_joint_limit_violation_rad": float(np.maximum(np.maximum(lower - positions[:, 7:], positions[:, 7:] - upper), 0).max()),
        "all_finite": bool(np.isfinite(positions).all()),
    }


def run(
    *,
    motion_path: Path,
    output_folder: Path,
    configuration: RetargetingConfiguration,
    frequency: int = 120,
    max_frames: int | None = None,
    preprocess: bool = True,
) -> None:
    source_path = REPOSITORY_PATH / "experiment/g1/environment/mjcf/g1.xml"
    robot_path = MICRODUCK_PATH / "environment/mjcf/microduck/robot_walk.xml"
    backlash_path = robot_path.with_name("robot_walk_backlash.xml")
    source_model = mujoco.MjModel.from_xml_path(str(source_path))
    model = mujoco.MjModel.from_xml_path(str(robot_path))
    backlash_model = mujoco.MjModel.from_xml_path(str(backlash_path))
    source_positions = read_motion(motion_path, source_model)[:max_frames]
    body_positions, body_quaternions, source_ground = prepare_targets(source_model, source_positions, configuration)
    maximum_target_lift = adapt_sole_clearance(model, body_positions, body_quaternions, configuration.ground_clearance_m)
    output_folder.mkdir(parents=True, exist_ok=True)
    config_path = output_folder / "gmr_configuration.json"
    config_path.write_text(json.dumps(gmr_configuration(model, configuration), indent=2) + "\n")
    positions = retarget(robot_path, config_path, configuration, body_positions, body_quaternions, frequency)
    maximum_ground_lift = align_to_ground(model, positions, configuration.ground_clearance_m)
    statistics = motion_statistics(model, positions, frequency)
    if not statistics["all_finite"] or statistics["maximum_joint_limit_violation_rad"] > 1e-5:
        raise ValueError(f"Retargeting produced invalid joint coordinates: {statistics}")
    name = motion_path.stem
    write_motion(output_folder / "rigid" / f"{name}.csv", model, positions)
    write_motion(output_folder / "backlash" / f"{name}.csv", backlash_model, with_zero_backlash(model, backlash_model, positions))
    statistics.update(
        source=str(motion_path.resolve()),
        source_ground_height_m=source_ground,
        maximum_ground_lift_m=maximum_ground_lift,
        maximum_target_sole_lift_m=maximum_target_lift,
        configuration=asdict(configuration),
    )
    (output_folder / "statistics.json").write_text(json.dumps(statistics, indent=2) + "\n")
    if preprocess:
        from robot_student.run.motion_preprocess import MotionPreprocess

        MotionPreprocess(
            robot_path=backlash_path,
            motion_folder=output_folder / "backlash",
            output_folder=output_folder / "preprocessed",
            motion_frequency=frequency,
            debug_level=logging.WARNING,
        ).run()
    print(f"Saved {name}: {len(positions)} frames at {frequency} Hz to {output_folder}", flush=True)


if __name__ == "__main__":
    run(
        motion_path=REPOSITORY_PATH / "experiment/g1/dataset/motion_decode/test/BG_Normal_Walking_00001.csv",
        output_folder=REPOSITORY_PATH / "result/microduck_motion",
        configuration=RetargetingConfiguration(),
        frequency=120,
        max_frames=None,
        preprocess=True,
    )
