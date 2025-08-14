#
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from typing import List, Optional

import numpy as np
import os  # Added to check display environment availability
import rerun as rr
import rerun.blueprint as rrb

import cuvslam as vslam

# Constants
DEFAULT_NUM_VIZ_CAMERAS = 1
POINT_RADIUS = 5.0
ARROW_SCALE = 0.1
GRAVITY_ARROW_SCALE = 0.02
CAMERA_BASELINE = 0.05  # Approximate baseline used when a second camera is present
FRUSTUM_DISTANCE = 0.1  # Distance from camera origin to image plane controlling frustum size


class RerunVisualizer:
    """Rerun-based visualizer for cuVSLAM tracking results."""
    
    def __init__(self, num_viz_cameras: int = DEFAULT_NUM_VIZ_CAMERAS) -> None:
        """Initialize rerun visualizer.
        
        Args:
            num_viz_cameras: Number of cameras to visualize
        """
        self.num_viz_cameras = num_viz_cameras
        # Only spawn a viewer when a display server is available.  This avoids
        # winit errors ("neither WAYLAND_DISPLAY nor DISPLAY is set") when
        # running in headless environments or inside containers without X/Wayland.
        can_spawn = bool(
            os.environ.get("DISPLAY")
            or os.environ.get("WAYLAND_DISPLAY")
            or os.environ.get("WAYLAND_SOCKET")
        )
        rr.init("cuVSLAM Visualizer", spawn=can_spawn)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
        
        if not can_spawn:
            # Inform the user how to open a viewer manually.
            print(
                "[RerunVisualizer] Display server not detected – the Rerun viewer "
                "was NOT spawned automatically. You can visualise the stream by "
                "starting a viewer manually, e.g. 'rerun viewer --connect'."
            )

        # Set up the visualization layout
        self._setup_blueprint()
        self.track_colors = {}
        self._pinhole_logged = set()  # Keep track of which cameras already have pinhole intrinsics logged

    def _setup_blueprint(self) -> None:
        """Set up the Rerun blueprint for visualization layout."""
        rr.send_blueprint(
            rrb.Blueprint(
                rrb.TimePanel(state="collapsed"),
                rrb.Horizontal(
                    column_shares=[0.5, 0.5],
                    contents=[
                        rrb.Vertical(contents=[
                            rrb.Spatial2DView(origin=f'world/camera_{i}')
                            for i in range(self.num_viz_cameras)
                        ]),
                        rrb.Spatial3DView(origin='world')
                    ]
                )
            ),
            make_active=True
        )

    def _log_rig_pose(
        self, rotation_quat: np.ndarray, translation: np.ndarray
    ) -> None:
        """Log rig pose and camera frustums to Rerun.
        
        Args:
            rotation_quat: Rotation quaternion
            translation: Translation vector
        """
        # Iterate over the number of cameras we are visualizing and log a transform
        for cam_idx in range(self.num_viz_cameras):
            cam_path = f"world/camera_{cam_idx}"

            # Offset successive cameras with a fixed baseline along +X so the two frustums are visible.
            cam_translation = translation + np.array([cam_idx * CAMERA_BASELINE, 0.0, 0.0])

            rr.log(
                cam_path,
                rr.Transform3D(
                    translation=cam_translation,
                    quaternion=rotation_quat,
                    # No additional scale – frustum size is controlled via image_plane_distance
                )
            )

            # Make sure a pinhole component is logged exactly once so Rerun can draw the frustum
            if cam_idx not in self._pinhole_logged:
                # Use a simple heuristic for intrinsics based on a 90° horizontal FOV if none are provided.
                # Width/height will be filled in later once we have the first image.
                rr.log(
                    cam_path,
                    rr.Pinhole(
                        focal_length=[1.0, 1.0],  # Placeholder values, updated when first image arrives
                        principal_point=[0.5, 0.5],
                        resolution=[1, 1],
                        image_plane_distance=FRUSTUM_DISTANCE,
                    ),
                    static=True
                )
                self._pinhole_logged.add(cam_idx)

    def _log_observations(
        self,
        observations_main_cam: List[vslam.Observation],
        image: np.ndarray,
        camera_name: str
    ) -> None:
        """Log 2D observations for a specific camera with consistent colors.
        
        Args:
            observations_main_cam: List of observations
            image: Camera image
            camera_name: Name of the camera for logging
        """
        # Handle different image datatypes for compression
        if image.dtype == np.uint8:
            image_log = rr.Image(image).compress()
        else:
            # For other datatypes, don't compress to avoid issues
            image_log = rr.Image(image)

        # Always log the image so 2D views are populated, even if there are no feature observations
        rr.log(f"world/{camera_name}", image_log)

        if not observations_main_cam:
            return

        # Assign random color to new tracks
        for obs in observations_main_cam:
            if obs.id not in self.track_colors:
                self.track_colors[obs.id] = np.random.randint(0, 256, size=3)

        points = np.array([[obs.u, obs.v] for obs in observations_main_cam])
        colors = np.array([
            self.track_colors[obs.id] for obs in observations_main_cam
        ])

        rr.log(
            f"world/{camera_name}/observations",
            rr.Points2D(positions=points, colors=colors, radii=POINT_RADIUS)
        )

    def _log_gravity(self, gravity: np.ndarray) -> None:
        """Log gravity vector to Rerun.
        
        Args:
            gravity: Gravity vector
        """
        rr.log(
            "world/camera_0/gravity",
            rr.Arrows3D(
                vectors=gravity,
                colors=[[255, 0, 0]],
                radii=GRAVITY_ARROW_SCALE
            )
        )

    def visualize_frame(
        self,
        frame_id: int,
        images: List[np.ndarray],
        pose: vslam.Pose,
        observations_main_cam: List[List[vslam.Observation]],
        trajectory: List[np.ndarray],
        timestamp: int,
        gravity: Optional[np.ndarray] = None
    ) -> None:
        """Visualize current frame state using Rerun.
        
        Args:
            frame_id: Current frame ID
            images: List of camera images
            pose: Current pose estimate
            observations_main_cam: List of observations for each camera
            trajectory: List of trajectory points
            timestamp: Current timestamp
            gravity: Optional gravity vector
        """
        rr.set_time_sequence("frame", frame_id)
        rr.log("world/trajectory", rr.LineStrips3D(trajectory), static=True)

        self._log_rig_pose(pose.rotation, pose.translation)
        
        # Update pinhole intrinsics with the real image size and focal length once we have data
        for i in range(self.num_viz_cameras):
            cam_path = f"world/camera_{i}"
            if i in self._pinhole_logged:
                # The pinhole may still have placeholder intrinsics; update with actual image info
                height, width = images[i].shape[:2]
                rr.log(
                    cam_path,
                    rr.Pinhole(
                        resolution=[width, height],
                        focal_length=[width / 2.0, width / 2.0],  # 90° HFOV approximation
                        principal_point=[width / 2.0, height / 2.0]
                        ,image_plane_distance=FRUSTUM_DISTANCE
                    ),
                    static=True
                )

        for i in range(self.num_viz_cameras):
            self._log_observations(
                observations_main_cam[i] if i < len(observations_main_cam) else [],
                images[i],
                f"camera_{i}"
            )
            
        if gravity is not None:
            self._log_gravity(gravity)
            
        rr.log("world/timestamp", rr.TextLog(str(timestamp)))
