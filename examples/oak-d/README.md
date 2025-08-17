# Tutorial: Running PyCuVSLAM Stereo Visual Odometry on OAK-D Stereo Camera

This tutorial demonstrates how to perform live PyCuVSLAM tracking using unrectified stereo images captured from an OAK-D stereo camera

> **Notes:**
> * The provided script has been developed and validated on the OAK-D W Pro stereo camera. Distortion models and order of cameras and its frames may differ for other OAK-D models. For more information about distortion models supported by cuVSLAM, see the [EuroC tutorial](../euroc/README.md#distortion-models).
> * **Global shutter** is a fundamental requirement for cuVSLAM. Please ensure your camera uses a global shutter sensor.

## Static Masks to Improve Visual Tracking

When using unrectified stereo images for visual tracking, residual geometric distortion may persist near the peripheral regions even after distortion correction. To mitigate this and improve tracking quality, we recommend applying static masks around the outer frame. These masks prevent PyCuVSLAM from selecting unreliable features near distorted image borders.

<img src="../../assets/tutorial_raw_images.png" alt="Static Masks Example" width="600" />

The figure above illustrates an example fisheye image from the [TUM-VI dataset](https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset). The original distorted fisheye image (left) and the corresponding undistorted image (right) are shown. The implemented static masks are indicated by red transparent borders. Each border (top, bottom, left, right) have independently specified thicknesses, allowing flexibility to mask out distorted regions appropriately.

To define these outer mask borders, specify pixel values independently for each camera instance as follows:

```
cam = vslam.Camera()
cam.border_top = 20
cam.border_bottom = 30
cam.border_left = 30
cam.border_right = 50
```

## Setting up the cuvslam environment
Refer to the [Installation Guide](../../README.md#pycuvslam-installation) for instructions on installing and configuring all required dependencies

## Setting up DepthAI
1. Install the [DepthAI Python library](https://github.com/luxonis/depthai-python) following the official documentation
2. Test your setup by running a basic [camera example](https://docs.luxonis.com/software/depthai/examples/rgb_preview/). Ensure it works correctly before proceeding

## Running Stereo Visual Odometry

```bash
python3 examples/oak-d/run_stereo.py
```

You should see the following interactive visualization in rerun: 
![Visualization Example](../../assets/tutorial_oakd_stereo.gif)

> **Note**: The PyCuVSLAM stereo tracker expects reliably synchronized stereo pairs with a stable FPS. If your camera pipeline is doing extensive on-device processing or AI inference, frame rates may drop, and image pairs may become unsynchronized. Watch for warnings about low FPS or mismatched stereo frames

If you experience low FPS even in the basic setup, you can investigate potential bottlenecks using the supplied [measurement tools](https://docs.luxonis.com/software/depthai/optimizing/) for OAK devices 

## Offline Playback (run_offline.py)

Use this when you have a dataset recorded previously (e.g. with `examples/oak-d/simple_recorder.py`) and want to replay it through PyCuVSLAM.

### Expected directory structure

```
run1/
  calibration.json           # camera & (optional) IMU calibration
  imu.csv                    # optional: accelerometer + gyroscope stream
  left.csv                   # mapping/index for left stream
  right.csv                  # mapping/index for right stream

  # Image mode (PNG, mono8)
  left/
    123456789.png
  right/
    123456789.png

  # Video mode (choose one of the following pairs)
  left.mp4                   # H.265 remuxed in MP4 container
  right.mp4
  # or
  left.h265                  # raw H.265 elementary stream
  right.h265
```

### CSV schemas and assumptions

- **left.csv/right.csv (image mode)**: header is `timestamp_ns,filename`
  - Example row: `1717000000123456,123456789.png`
- **left.csv/right.csv (video mode)**: header is `ts_ns,frame_idx`
  - Example row: `1717000000123456,42`
  - Frames are decoded sequentially and paired by timestamp within a 5 ms tolerance.
- **imu.csv (optional)**: header is `ts_ns,gyro_x,gyro_y,gyro_z,accel_x,accel_y,accel_z`
  - Units should be consistent with your calibration (typically rad/s for gyro, m/s^2 for accel).

### calibration.json expectations

- `left` and `right` camera entries must include:
  - `intrinsics`: 3×3 matrix; `intrinsics[0][0]=fx`, `intrinsics[1][1]=fy`, `intrinsics[0][2]=cx`, `intrinsics[1][2]=cy`
  - `distortion`: polynomial model coefficients array
  - `resolution`: `[width, height]`
  - `extrinsics`: 4×4 transform from camera to rig, in centimetres (converted to metres at runtime)
- Optional `imu` entry (for `--with-imu`):
  - `rig_from_imu`: 4×4 transform in centimetres
  - `gyroscope_noise_density`, `gyroscope_random_walk`
  - `accelerometer_noise_density`, `accelerometer_random_walk`
  - `frequency`

### Run the offline playback

- Minimal (stereo-only):

```bash
pixi run examples/oak-d/run_offline.py --data ~/ds3 --horizontal-stereo-camera --save-rrd out.rrd
```

- With IMU (stereo-VIO), if `imu.csv` and IMU calibration are present:

```bash
pixi run examples/oak-d/run_offline.py --data ~/ds3 --with-imu --horizontal-stereo-camera --save-rrd out.rrd
```

Notes:
- Use `--horizontal-stereo-camera` when the left/right cameras form a horizontal stereo pair (typical OAK-D setup).
- For video mode, ensure `left.mp4/right.mp4` (or `left.h265/right.h265`) exist alongside `left.csv/right.csv` with `ts_ns,frame_idx` headers.
- The first 60 frames are used as warmup and are not processed.
- Add `--save-rrd <file.rrd>` to save a full Rerun recording for later inspection.