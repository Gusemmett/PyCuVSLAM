"""Offline playback for datasets recorded with simple_recorder.py.

This script replays a previously recorded OAK-D dataset in either stereo
visual odometry (VO) or stereo-VIO (visual-inertial odometry) mode using
PyCuVSLAM.  It consumes the folder structure created by
`examples/oak-d/simple_recorder.py`:

    run1/
      calibration.json           # camera & IMU calibration
      imu.csv                    # raw accelerometer + gyroscope (optional)
      left.csv                   # image mode: timestamp_ns, filename
                                 # video mode: ts_ns, frame_idx
      right.csv                  # same schema as left.csv
      left/123456789.png         # left images as PNG (mono8)   [image mode]
      right/123456789.png        # right images as PNG (mono8)  [image mode]
      left.mp4                   # H.265 remuxed in MP4 container [video mode]
      right.mp4                  # H.265 remuxed in MP4 container [video mode]
      (alternatively)
      left.h265                  # raw H.265 elementary stream    [video mode]
      right.h265                 # raw H.265 elementary stream    [video mode]

By default the script operates in **stereo-only** mode.  Pass `--with-imu`
(or `--mode svio`) to additionally feed the recorded IMU stream and enable
inertial tracking.

Example usages
--------------
Stereo only (visual odometry):
    python3 run_offline.py --data ./run1
Stereo + IMU (stereo-VIO):
    python3 run_offline.py --data ./run1 --with-imu
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import pathlib
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

import cuvslam as vslam

# -----------------------------------------------------------------------------
# Configure Python logging – helpful to debug file/stream issues
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Add visualiser helper from the RealSense examples (same folder layout as the
# live OAK-D scripts)
CUR_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CUR_DIR / "../realsense"))
from visualizer import RerunVisualizer  # pylint: disable=wrong-import-position

# -----------------------------------------------------------------------------
# Constants  (keep in sync with live OAK-D examples for comparable behaviour)
# -----------------------------------------------------------------------------
WARMUP_FRAMES = 60
SYNC_THRESHOLD_NS = 5 * 1_000_000   # 5 ms in nanoseconds
IMAGE_JITTER_THRESHOLD_NS = 35 * 1_000_000  # 35 ms in nanoseconds

# Static border masks – helps tracking on unrectified OAK-D fisheye images
BORDER_TOP = 50
BORDER_BOTTOM = 0
BORDER_LEFT = 70
BORDER_RIGHT = 70

# DepthAI calibration is in centimetres – convert to metres for cuVSLAM
CM_TO_METERS = 100.0

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def oak_transform_to_pose(oak_extrinsics: List[List[float]]) -> vslam.Pose:
    """Convert a 4×4 extrinsics matrix (centimetres) to a cuVSLAM Pose."""
    extr = np.asarray(oak_extrinsics)
    rot_m = extr[:3, :3]
    trans_m = extr[:3, 3] / CM_TO_METERS  # cm –> m
    quat = Rotation.from_matrix(rot_m).as_quat()
    return vslam.Pose(rotation=quat, translation=trans_m)


def set_cuvslam_camera(oak_params: Dict[str, Any]) -> vslam.Camera:
    """Build a vslam.Camera from calibration parameters stored in JSON."""
    cam = vslam.Camera()

    cam.distortion = vslam.Distortion(
        vslam.Distortion.Model.Polynomial, oak_params["distortion"]
    )

    cam.focal = (
        oak_params["intrinsics"][0][0],  # fx
        oak_params["intrinsics"][1][1],  # fy
    )
    cam.principal = (
        oak_params["intrinsics"][0][2],  # cx
        oak_params["intrinsics"][1][2],  # cy
    )
    cam.size = oak_params["resolution"]
    cam.rig_from_camera = oak_transform_to_pose(oak_params["extrinsics"])

    # Mask out border regions subject to high distortion
    cam.border_top = BORDER_TOP
    cam.border_bottom = BORDER_BOTTOM
    cam.border_left = BORDER_LEFT
    cam.border_right = BORDER_RIGHT

    return cam


def load_csv_mapping(csv_path: pathlib.Path) -> List[Tuple[int, str]]:
    """Load `<timestamp_ns>, <filename>` mapping CSV produced by recorder."""
    mapping: List[Tuple[int, str]] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if not row:
                continue
            ts_ns = int(row[0])
            fname = row[1]
            mapping.append((ts_ns, fname))
    return mapping


def load_stream_csv(csv_path: pathlib.Path) -> List[Tuple[int, int]]:
    """Load `ts_ns,frame_idx` mapping CSV produced when recording video.

    Returns a list of (timestamp_ns, frame_index) tuples.
    """
    mapping: List[Tuple[int, int]] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if not row:
                continue
            ts_ns = int(row[0])
            frame_idx = int(row[1])
            mapping.append((ts_ns, frame_idx))
    return mapping


class StereoVideoReader:
    """Decode paired left/right video streams (MP4 or H.265) and align by timestamps.

    Frames are decoded sequentially from both streams. The reader advances the
    earlier-timestamp stream until the absolute timestamp difference is within
    SYNC_THRESHOLD_NS, then returns the aligned pair as mono8 images.
    """

    def __init__(
        self,
        left_video: pathlib.Path,
        right_video: pathlib.Path,
        left_map: List[Tuple[int, int]],
        right_map: List[Tuple[int, int]],
    ) -> None:
        self.cap_left = cv2.VideoCapture(str(left_video))
        self.cap_right = cv2.VideoCapture(str(right_video))
        self.left_map = left_map
        self.right_map = right_map
        self.i = 0
        self.j = 0

        if not self.cap_left.isOpened() or not self.cap_right.isOpened():
            raise RuntimeError("Failed to open video files for playback")

    def _read_one(self, which: str) -> Optional[np.ndarray]:
        cap = self.cap_left if which == "left" else self.cap_right
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def read_aligned(self) -> Optional[Tuple[int, np.ndarray, np.ndarray]]:
        while self.i < len(self.left_map) and self.j < len(self.right_map):
            ts_left = self.left_map[self.i][0]
            ts_right = self.right_map[self.j][0]
            diff = abs(ts_left - ts_right)

            if diff <= SYNC_THRESHOLD_NS:
                left_img = self._read_one("left")
                right_img = self._read_one("right")
                if left_img is None or right_img is None:
                    return None
                self.i += 1
                self.j += 1
                return ts_left, left_img, right_img

            if ts_left < ts_right:
                if self._read_one("left") is None:
                    return None
                self.i += 1
            else:
                if self._read_one("right") is None:
                    return None
                self.j += 1

        return None


def load_imu_stream(csv_path: pathlib.Path) -> List[vslam.ImuMeasurement]:
    """Load IMU CSV and convert each row to vslam.ImuMeasurement."""
    stream: List[vslam.ImuMeasurement] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            if not row:
                continue
            ts_ns = int(row[0])
            gyro = np.array(list(map(float, row[1:4])))
            accel = np.array(list(map(float, row[4:7])))
            meas = vslam.ImuMeasurement()
            meas.timestamp_ns = ts_ns
            meas.angular_velocities = gyro
            meas.linear_accelerations = accel
            stream.append(meas)
    return stream


# -----------------------------------------------------------------------------
# Main playback routine
# -----------------------------------------------------------------------------

def main() -> None:  # noqa: C901 – main is slightly long but self-contained
    import argparse
    parser = argparse.ArgumentParser(
        description="Offline replay of OAK-D dataset for PyCuVSLAM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=pathlib.Path,
        required=True,
        help="Path to dataset folder (recorded by simple_recorder.py)",
    )
    parser.add_argument(
        "--with-imu",
        action="store_true",
        help="Enable stereo-VIO by feeding the recorded IMU stream.",
    )
    parser.add_argument(
        "--save-rrd",
        type=pathlib.Path,
        default=None,
        metavar="FILE",
        help="Save a full Rerun recording of the playback to this .rrd file.",
    )
    parser.add_argument(
        "--horizontal-stereo-camera",
        action="store_true",
        help="Set this if the left/right cameras form a horizontal stereo pair.",
    )
    args = parser.parse_args()

    dataset_dir = args.data.resolve()
    if not dataset_dir.exists():
        parser.error(f"Dataset path {dataset_dir} does not exist.")

    # ------------------------- Load calibration -----------------------------
    calib_path = dataset_dir / "calibration.json"
    if not calib_path.exists():
        parser.error("calibration.json not found in dataset – please check path.")

    calib_json: Dict[str, Any] = json.loads(calib_path.read_text())

    # Cameras
    cameras = [
        set_cuvslam_camera(calib_json["left"]),
        set_cuvslam_camera(calib_json["right"]),
    ]

    rig = vslam.Rig()
    rig.cameras = cameras

    # Optional IMU
    use_imu = args.with_imu and (dataset_dir / "imu.csv").exists()
    if args.with_imu and not use_imu:
        logger.warning("--with-imu specified but imu.csv missing – continuing in stereo-only mode")

    if use_imu:
        imu_cfg: Dict[str, Any] = calib_json["imu"]
        imu_cal = vslam.ImuCalibration()
        imu_cal.rig_from_imu = oak_transform_to_pose(imu_cfg["rig_from_imu"])
        imu_cal.gyroscope_noise_density = imu_cfg["gyroscope_noise_density"]
        imu_cal.gyroscope_random_walk = imu_cfg["gyroscope_random_walk"]
        imu_cal.accelerometer_noise_density = imu_cfg["accelerometer_noise_density"]
        imu_cal.accelerometer_random_walk = imu_cfg["accelerometer_random_walk"]
        imu_cal.frequency = imu_cfg["frequency"]
        rig.imus = [imu_cal]

    # ------------------------- Tracker configuration -----------------------
    cfg = vslam.Tracker.OdometryConfig(
        async_sba=False,
        enable_final_landmarks_export=True,
        enable_observations_export=True,
        odometry_mode=(
            vslam.Tracker.OdometryMode.Inertial if use_imu else vslam.Tracker.OdometryMode.Multicamera
        ),
        horizontal_stereo_camera=args.horizontal_stereo_camera,
    )

    tracker = vslam.Tracker(rig, cfg)
    logger.info("Tracker initialised – mode: %s", "SVIO" if use_imu else "Stereo")

    # ------------------------- Visualisation -------------------------------
    visualizer = RerunVisualizer(num_viz_cameras=2)

    # ------------------------- Load streams --------------------------------
    left_csv_path = dataset_dir / "left.csv"
    right_csv_path = dataset_dir / "right.csv"
    # Prefer MP4 if present; otherwise fall back to raw H.265 elementary streams
    left_mp4_path = dataset_dir / "left.mp4"
    right_mp4_path = dataset_dir / "right.mp4"
    left_h265_path = dataset_dir / "left.h265"
    right_h265_path = dataset_dir / "right.h265"

    video_left_path: Optional[pathlib.Path] = None
    video_right_path: Optional[pathlib.Path] = None
    if left_mp4_path.exists() and right_mp4_path.exists():
        video_left_path = left_mp4_path
        video_right_path = right_mp4_path
    elif left_h265_path.exists() and right_h265_path.exists():
        video_left_path = left_h265_path
        video_right_path = right_h265_path

    use_video = video_left_path is not None and video_right_path is not None

    if use_video:
        left_stream = load_stream_csv(left_csv_path)
        right_stream = load_stream_csv(right_csv_path)
        if not left_stream or not right_stream:
            parser.error("left.csv/right.csv missing or empty for video playback")
        video_reader = StereoVideoReader(video_left_path, video_right_path, left_stream, right_stream)
    else:
        left_map = load_csv_mapping(left_csv_path)
        right_map = load_csv_mapping(right_csv_path)
        if len(left_map) != len(right_map):
            logger.warning("Left/right CSV differ in length (%d vs %d)", len(left_map), len(right_map))
        right_dict = {ts: fname for ts, fname in right_map}

    # Load IMU if requested
    imu_stream: List[vslam.ImuMeasurement] = []
    if use_imu:
        imu_stream = load_imu_stream(dataset_dir / "imu.csv")

    imu_idx = 0
    frame_id = 0
    trajectory: List[np.ndarray] = []

    if use_video:
        prev_ts: Optional[int] = None
        while True:
            aligned = video_reader.read_aligned()
            if aligned is None:
                break
            ts_left, left_img, right_img = aligned

            if use_imu:
                while imu_idx < len(imu_stream) and imu_stream[imu_idx].timestamp_ns <= ts_left:
                    tracker.register_imu_measurement(0, imu_stream[imu_idx])
                    imu_idx += 1

            if prev_ts is not None:
                jitter = ts_left - prev_ts
                if jitter > IMAGE_JITTER_THRESHOLD_NS:
                    logger.warning(
                        "Camera stream message drop: timestamp gap (%.2f ms) exceeds threshold %.2f ms",
                        jitter / 1e6,
                        IMAGE_JITTER_THRESHOLD_NS / 1e6,
                    )
            prev_ts = ts_left

            frame_id += 1
            if frame_id <= WARMUP_FRAMES:
                continue

            odom_est, _ = tracker.track(ts_left, (left_img, right_img))
            odom_pose = odom_est.world_from_rig.pose
            trajectory.append(odom_pose.translation)

            gravity = (
                tracker.get_last_gravity() if use_imu and hasattr(tracker, "get_last_gravity") else None
            )

            visualizer.visualize_frame(
                frame_id=frame_id,
                images=[left_img, right_img],
                pose=odom_pose,
                observations_main_cam=[
                    tracker.get_last_observations(0),
                    tracker.get_last_observations(1),
                ],
                trajectory=trajectory,
                timestamp=ts_left,
                gravity=gravity,
            )
    else:
        for ts_left, left_fname in left_map:
            right_fname = right_dict.get(ts_left)
            if right_fname is None:
                logger.warning("Right frame missing for timestamp %d – skipping", ts_left)
                continue

            if use_imu:
                while imu_idx < len(imu_stream) and imu_stream[imu_idx].timestamp_ns <= ts_left:
                    tracker.register_imu_measurement(0, imu_stream[imu_idx])
                    imu_idx += 1

            left_img_path = dataset_dir / "left" / left_fname
            right_img_path = dataset_dir / "right" / right_fname
            if not left_img_path.exists() or not right_img_path.exists():
                logger.warning("Image files missing for timestamp %d – skipping", ts_left)
                continue

            left_img = cv2.imread(str(left_img_path), cv2.IMREAD_GRAYSCALE)
            right_img = cv2.imread(str(right_img_path), cv2.IMREAD_GRAYSCALE)
            if left_img is None or right_img is None:
                logger.warning("Failed to read image files for timestamp %d – skipping", ts_left)
                continue

            frame_id += 1
            if frame_id <= WARMUP_FRAMES:
                continue

            odom_est, _ = tracker.track(ts_left, (left_img, right_img))
            odom_pose = odom_est.world_from_rig.pose
            trajectory.append(odom_pose.translation)

            gravity = (
                tracker.get_last_gravity() if use_imu and hasattr(tracker, "get_last_gravity") else None
            )

            visualizer.visualize_frame(
                frame_id=frame_id,
                images=[left_img, right_img],
                pose=odom_pose,
                observations_main_cam=[
                    tracker.get_last_observations(0),
                    tracker.get_last_observations(1),
                ],
                trajectory=trajectory,
                timestamp=ts_left,
                gravity=gravity,
            )

    logger.info("Playback finished – processed %d frames", frame_id)

    # ---------------------------------------------------------------------
    # Save full Rerun recording (images, poses, observations, …) if requested
    # ---------------------------------------------------------------------
    if args.save_rrd is not None:
        try:
            import rerun as rr  # pylint: disable=import-error

            rr.save(str(args.save_rrd))
            logger.info("Rerun recording saved to %s", args.save_rrd.resolve())
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Failed to save Rerun recording: %s", exc)


if __name__ == "__main__":
    main()
