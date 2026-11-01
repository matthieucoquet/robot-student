"""Adapt a G1 motion CSV to Microduck using stock GMR and fixed frame calibration."""

import argparse
import csv
import json
import logging
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

REPOSITORY_PATH = Path(__file__).resolve().parents[2]
ROBOT_PATH = Path(__file__).resolve().parent / "environment/mjcf/microduck/robot_walk.xml"
SOURCE_ROBOT_PATH = REPOSITORY_PATH / "experiment/g1/environment/mjcf/g1.xml"
ROOT_COLUMNS = ("root_pos_x(m)", "root_pos_y(m)", "root_pos_z(m)", "root_rot_w", "root_rot_x", "root_rot_y", "root_rot_z")
FOOT_NAMES = ("left_foot", "right_foot")
BODY_NAMES = ("pelvis", *FOOT_NAMES, "torso_link")
HEAD_JOINT_NAMES = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")


def joint_names(model: mujoco.MjModel) -> list[str]:
    if model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE or np.any(model.jnt_type[1:] != mujoco.mjtJoint.mjJNT_HINGE):
        raise ValueError("Expected one free root followed by hinge joints")
    return [model.joint(index).name for index in range(1, model.njnt)]


def read_motion(path: Path, model: mujoco.MjModel) -> np.ndarray:
    expected_columns = [*ROOT_COLUMNS, *(f"dof_{name}(rad)" for name in joint_names(model))]
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        columns = [column.strip() for column in next(reader, [])]
        if len(set(columns)) != len(columns) or set(columns) != set(expected_columns):
            raise ValueError("Motion CSV must contain the named root and joint coordinates of the source robot")
        rows = np.asarray(list(reader), dtype=np.float64)
    if rows.ndim != 2 or len(rows) < 2 or rows.shape[1] != len(columns) or not np.isfinite(rows).all():
        raise ValueError("Motion CSV must contain at least two finite frames")
    positions = rows[:, [columns.index(name) for name in expected_columns]]
    lengths = np.linalg.norm(positions[:, 3:7], axis=1, keepdims=True)
    if (lengths < 1e-8).any():
        raise ValueError("Motion CSV contains a zero root quaternion")
    positions[:, 3:7] /= lengths
    return positions


def source_body_poses(model: mujoco.MjModel, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Extract pelvis, soles, and torso; remove only one fixed origin, heading, and ground height."""
    data = mujoco.MjData(model)
    body_positions = np.empty((len(positions), len(BODY_NAMES), 3))
    body_quaternions = np.empty((len(positions), len(BODY_NAMES), 4))
    torso_index = model.body("torso_link").id
    initial_heading = Rotation.from_quat(positions[0, 3:7], scalar_first=True).as_euler("xyz")[2]
    heading_inverse = Rotation.from_euler("z", -initial_heading)
    origin = np.array([*positions[0, :2], 0.0])
    for frame_index, frame in enumerate(positions):
        data.qpos[:] = frame
        mujoco.mj_kinematics(model, data)
        body_positions[frame_index, 0] = frame[:3]
        rotations = [Rotation.from_quat(frame[3:7], scalar_first=True)]
        for body_index, name in enumerate(FOOT_NAMES, start=1):
            site_index = model.site(name).id
            body_positions[frame_index, body_index] = data.site_xpos[site_index]
            rotations.append(Rotation.from_matrix(data.site_xmat[site_index].reshape(3, 3)))
        body_positions[frame_index, -1] = data.xpos[torso_index]
        rotations.append(Rotation.from_matrix(data.xmat[torso_index].reshape(3, 3)))
        body_positions[frame_index] = heading_inverse.apply(body_positions[frame_index] - origin)
        body_quaternions[frame_index] = [(heading_inverse * rotation).as_quat(scalar_first=True) for rotation in rotations]
    source_ground = float(body_positions[:, 1:3, 2].min())
    body_positions[:, :, 2] -= source_ground
    return body_positions, body_quaternions, source_ground


def gmr_configuration(source_model: mujoco.MjModel, target_model: mujoco.MjModel, motion_scale: float) -> dict:
    """Configure native GMR scaling, sole frame/stance offsets, and neutral torso-to-head orientation calibration."""
    source_data, target_data = mujoco.MjData(source_model), mujoco.MjData(target_model)
    mujoco.mj_kinematics(source_model, source_data)
    mujoco.mj_kinematics(target_model, target_data)
    matches = {"trunk_base": ["pelvis", 100.0, 5.0, [0, 0, 0], [1, 0, 0, 0]]}
    for name in FOOT_NAMES:
        site = target_model.site(name)
        body_index = site.bodyid[0]
        lateral_offset = target_data.site_xpos[site.id, 1] - motion_scale * source_data.site_xpos[source_model.site(name).id, 1]
        # GMR applies position offsets in the already-corrected target body frame.
        body_rotation = target_data.xmat[body_index].reshape(3, 3)
        position_offset = -site.pos + body_rotation.T @ np.array([0.0, lateral_offset, 0.0])
        matches[target_model.body(body_index).name] = [
            name,
            200.0,
            5.0,
            position_offset.tolist(),
            Rotation.from_quat(site.quat, scalar_first=True).inv().as_quat(scalar_first=True).tolist(),
        ]
    # Calibrate the neutral body frames once; track torso orientation without imposing G1 torso height on the head.
    source_torso_rotation = Rotation.from_matrix(source_data.xmat[source_model.body("torso_link").id].reshape(3, 3))
    target_head_rotation = Rotation.from_matrix(target_data.xmat[target_model.body("jaw_soft").id].reshape(3, 3))
    matches["jaw_soft"] = [
        "torso_link",
        0.0,
        5.0,
        [0, 0, 0],
        (source_torso_rotation.inv() * target_head_rotation).as_quat(scalar_first=True).tolist(),
    ]
    return {
        "human_height_assumption": 1.0,
        "human_root_name": "pelvis",
        "robot_root_name": "trunk_base",
        "human_scale_table": dict.fromkeys(BODY_NAMES, motion_scale),
        "ground_height": 0.0,
        "use_ik_match_table1": True,
        "use_ik_match_table2": False,
        "ik_match_table1": matches,
        "ik_match_table2": {},
    }


def retarget(config_path: Path, body_positions: np.ndarray, body_quaternions: np.ndarray, frequency: int) -> np.ndarray:
    from general_motion_retargeting import GeneralMotionRetargeting
    from general_motion_retargeting.params import IK_CONFIG_DICT, ROBOT_XML_DICT

    ROBOT_XML_DICT["microduck"] = ROBOT_PATH
    IK_CONFIG_DICT.setdefault("g1", {})["microduck"] = config_path.resolve()
    retargeter = GeneralMotionRetargeting(src_human="g1", tgt_robot="microduck", verbose=False, use_velocity_limit=True)
    retargeter.model.opt.timestep = 1.0 / frequency
    # Select the forward knee-bend branch using GMR's existing ConfigurationLimit; the joint axes are mirrored.
    joint_limits = retargeter.ik_limits[0]
    joint_limits.lower[retargeter.model.joint("left_knee").qposadr[0]] = 0.0
    joint_limits.upper[retargeter.model.joint("right_knee").qposadr[0]] = 0.0
    # Keep GMR's initialization, tasks, iteration loop, damping, and all other joint bounds unchanged.
    output = np.empty((len(body_positions), retargeter.model.nq))
    for frame_index, (positions, quaternions) in enumerate(zip(body_positions, body_quaternions, strict=True)):
        targets = dict(zip(BODY_NAMES, zip(positions, quaternions, strict=True), strict=True))
        output[frame_index] = retargeter.retarget(targets)
        if (frame_index + 1) % 240 == 0:
            print(f"Retargeted {frame_index + 1}/{len(output)} frames", flush=True)
    return output


def write_motion(path: Path, model: mujoco.MjModel, positions: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ",".join([*ROOT_COLUMNS, *(f"dof_{name}(rad)" for name in joint_names(model))])
    np.savetxt(path, positions, delimiter=",", header=header, comments="", fmt="%.10g")


def with_zero_backlash(model: mujoco.MjModel, backlash_model: mujoco.MjModel, positions: np.ndarray) -> np.ndarray:
    output = np.zeros((len(positions), backlash_model.nq))
    output[:, :7] = positions[:, :7]
    for name in joint_names(backlash_model):
        if not name.startswith("passive_"):
            output[:, backlash_model.joint(name).qposadr[0]] = positions[:, model.joint(name).qposadr[0]]
    return output


def motion_statistics(model: mujoco.MjModel, positions: np.ndarray, frequency: int) -> dict:
    data = mujoco.MjData(model)
    minimum_sole_height = np.inf
    geometry = []
    for name in ("left_foot_collision", "right_foot_collision"):
        geom = model.geom(name)
        mesh_index = geom.dataid[0]
        start = model.mesh_vertadr[mesh_index]
        geometry.append((geom.id, model.mesh_vert[start : start + model.mesh_vertnum[mesh_index]]))
    for frame in positions:
        data.qpos[:] = frame
        mujoco.mj_kinematics(model, data)
        for index, vertices in geometry:
            minimum_sole_height = min(
                minimum_sole_height, float((vertices @ data.geom_xmat[index].reshape(3, 3)[2]).min() + data.geom_xpos[index, 2])
            )
    violation = np.maximum(model.jnt_range[1:, 0] - positions[:, 7:], positions[:, 7:] - model.jnt_range[1:, 1])
    left_knee = positions[:, model.joint("left_knee").qposadr[0]]
    right_knee = positions[:, model.joint("right_knee").qposadr[0]]
    head_joint_ranges = {}
    for name in HEAD_JOINT_NAMES:
        joint_positions = positions[:, model.joint(name).qposadr[0]]
        head_joint_ranges[name] = [float(joint_positions.min()), float(joint_positions.max())]
    return {
        "frame_count": len(positions),
        "frequency": frequency,
        "duration_seconds": (len(positions) - 1) / frequency,
        "minimum_sole_height_m": minimum_sole_height,
        "maximum_sampled_joint_speed_rad_per_second": float(np.abs(np.diff(positions[:, 7:], axis=0)).max() * frequency),
        "maximum_joint_limit_violation_rad": float(np.maximum(violation, 0).max()),
        "left_knee_range_rad": [float(left_knee.min()), float(left_knee.max())],
        "right_knee_range_rad": [float(right_knee.min()), float(right_knee.max())],
        "maximum_backward_knee_bend_rad": float(max(0, -left_knee.min(), right_knee.max())),
        "head_joint_ranges_rad": head_joint_ranges,
        "all_finite": bool(np.isfinite(positions).all()),
    }


def run(
    *, motion_path: Path, output_folder: Path, motion_scale: float, frequency: int, max_frames: int | None, preprocess: bool, preview: bool
):
    if not np.isfinite(motion_scale) or motion_scale <= 0 or frequency <= 0:
        raise ValueError("Motion scale and frequency must be positive")
    if max_frames is not None and max_frames < 2:
        raise ValueError("At least two frames are required")
    if preview and not preprocess:
        raise ValueError("Preview recording requires preprocessing")
    source_model = mujoco.MjModel.from_xml_path(str(SOURCE_ROBOT_PATH))
    model = mujoco.MjModel.from_xml_path(str(ROBOT_PATH))
    source_positions = read_motion(motion_path, source_model)[:max_frames]
    body_positions, body_quaternions, source_ground = source_body_poses(source_model, source_positions)
    output_folder.mkdir(parents=True, exist_ok=True)
    config_path = output_folder / "gmr_configuration.json"
    config_path.write_text(json.dumps(gmr_configuration(source_model, model, motion_scale), indent=2) + "\n")
    positions = retarget(config_path, body_positions, body_quaternions, frequency)
    statistics = motion_statistics(model, positions, frequency)
    if (
        not statistics["all_finite"]
        or statistics["maximum_joint_limit_violation_rad"] > 1e-5
        or statistics["maximum_backward_knee_bend_rad"] > 1e-5
    ):
        raise ValueError(f"GMR produced invalid coordinates: {statistics}")
    name = motion_path.stem
    write_motion(output_folder / "rigid" / f"{name}.csv", model, positions)
    backlash_path = ROBOT_PATH.with_name("robot_walk_backlash.xml")
    backlash_model = mujoco.MjModel.from_xml_path(str(backlash_path))
    write_motion(output_folder / "backlash" / f"{name}.csv", backlash_model, with_zero_backlash(model, backlash_model, positions))
    statistics.update(source=str(motion_path.resolve()), motion_scale=motion_scale, source_ground_height_m=source_ground)
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
    if preview:
        from robot_student.run import MotionClipPlayer

        MotionClipPlayer(
            robot_path=backlash_path,
            motion_path=output_folder / "preprocessed" / f"{name}.pt",
            headless=True,
            debug_level=logging.WARNING,
            recording_path=output_folder / "preview.mp4",
            recording_camera_position=(0.45, 0.0, 0.05),
        ).run()
    print(f"Saved {name}: {len(positions)} frames at {frequency} Hz to {output_folder}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--motion-path", type=Path, default=REPOSITORY_PATH / "experiment/g1/dataset/motion_decode/test/BG_Normal_Walking_00001.csv"
    )
    parser.add_argument("--output-folder", type=Path, default=REPOSITORY_PATH / "result/microduck_motion_gmr")
    parser.add_argument("--motion-scale", type=float, default=0.15)
    parser.add_argument("--frequency", type=int, default=120, help="Input CSV sampling frequency")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--no-preprocess", action="store_true")
    parser.add_argument("--preview", action="store_true")
    arguments = parser.parse_args()
    run(
        motion_path=arguments.motion_path,
        output_folder=arguments.output_folder,
        motion_scale=arguments.motion_scale,
        frequency=arguments.frequency,
        max_frames=arguments.max_frames,
        preprocess=not arguments.no_preprocess,
        preview=arguments.preview,
    )
