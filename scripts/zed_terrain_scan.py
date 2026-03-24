"""
ZED 2i → terrain_scan[35] pipeline for AWM real-world deployment.

Replicates the sim ray caster: 7x5 grid at 0.15m resolution,
covering 0.9m wide x 0.6m deep ahead of the robot.

Output: terrain_scan[35] — relative heights in robot body frame,
        matching the observation the policy was trained on.

Standalone test:
    python3 zed_terrain_scan.py

Import usage:
    from zed_terrain_scan import TerrainScanner
    scanner = TerrainScanner()
    scanner.start()
    scan = scanner.get_terrain_scan(robot_base_z=0.0)
    gravity, yaw_rate = scanner.get_imu_data()
    scanner.stop()
"""

import numpy as np
import pyzed.sl as sl


# ---------------------------------------------------------------------------
# Grid parameters — must match sim ray caster exactly
# ---------------------------------------------------------------------------
GRID_NX = 7           # columns along forward (x) axis
GRID_NY = 5           # rows along lateral (y) axis
GRID_RES = 0.15       # metres between grid points
GRID_X_START = 0.15   # distance of first column ahead (metres)
GRID_Y_START = -0.30  # lateral offset of first row (metres, negative = right)
SEARCH_RADIUS = 0.07  # search radius per grid cell (half of GRID_RES)

# Pre-computed grid positions in robot body frame
GRID_X = np.array([GRID_X_START + i * GRID_RES for i in range(GRID_NX)])  # [0.15 .. 1.05]
GRID_Y = np.array([GRID_Y_START + j * GRID_RES for j in range(GRID_NY)])  # [-0.30 .. 0.30]


class TerrainScanner:
    """
    Wraps ZED 2i SDK 5.x to produce terrain_scan[35] and IMU data.

    Camera mounting parameters (adjust to match your physical setup):
        cam_height:  metres above robot base z
        cam_forward: metres forward of robot base origin
        tilt_deg:    downward tilt angle in degrees
    """

    def __init__(
        self,
        cam_height: float = 0.15,
        cam_forward: float = 0.05,
        tilt_deg: float = 30.0,
        resolution=sl.RESOLUTION.HD720,
        depth_mode=sl.DEPTH_MODE.ULTRA,
    ):
        self._camera = sl.Camera()

        init_params = sl.InitParameters()
        init_params.camera_resolution = resolution
        init_params.depth_mode = depth_mode
        init_params.coordinate_units = sl.UNIT.METER
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
        init_params.depth_minimum_distance = 0.3
        init_params.depth_maximum_distance = 2.0
        self._init_params = init_params

        self._runtime = sl.RuntimeParameters()
        self._runtime.confidence_threshold = 80

        self._point_cloud = sl.Mat()
        self._sensors_data = sl.SensorsData()

        # Camera-to-body extrinsic transform (4x4)
        self.T_cam_to_body = self._build_extrinsics(cam_height, cam_forward, tilt_deg)

        self._last_scan = np.zeros(GRID_NX * GRID_NY, dtype=np.float32)

    @staticmethod
    def _build_extrinsics(cam_height: float, cam_forward: float, tilt_deg: float) -> np.ndarray:
        """4x4 transform from camera frame to robot body frame."""
        tilt = np.radians(tilt_deg)
        R = np.array([
            [ np.cos(tilt), 0, np.sin(tilt)],
            [            0, 1,            0],
            [-np.sin(tilt), 0, np.cos(tilt)],
        ])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3,  3] = np.array([cam_forward, 0.0, cam_height])
        return T

    def start(self):
        err = self._camera.open(self._init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED 2i failed to open: {err}")
        print("[TerrainScanner] ZED 2i opened successfully")

    def stop(self):
        self._camera.close()
        print("[TerrainScanner] ZED 2i closed")

    def get_terrain_scan(self, robot_base_z: float = 0.0) -> np.ndarray:
        """
        Capture one frame and return terrain_scan[35].

        Args:
            robot_base_z: robot base height in world frame (metres).
                          Use 0.0 if unknown — heights will be relative
                          to the camera mounting height instead.

        Returns:
            np.ndarray shape (35,) — height of ground relative to robot base z.
            Positive = obstacle/step above base. Negative = drop below base.
            Zero = no valid depth at that grid cell.
        """
        if self._camera.grab(self._runtime) != sl.ERROR_CODE.SUCCESS:
            return self._last_scan

        self._camera.retrieve_measure(self._point_cloud, sl.MEASURE.XYZRGBA)
        pts_raw = self._point_cloud.get_data()[:, :, :3]  # (H, W, 3)

        # Flatten and remove invalid (NaN/inf) points
        pts = pts_raw.reshape(-1, 3)
        valid = np.isfinite(pts).all(axis=1)
        pts = pts[valid]

        if len(pts) == 0:
            return self._last_scan

        # Transform from camera frame → robot body frame
        pts_body = (self.T_cam_to_body[:3, :3] @ pts.T).T + self.T_cam_to_body[:3, 3]

        # Sample minimum height at each of the 35 grid cells
        scan = np.zeros(GRID_NX * GRID_NY, dtype=np.float32)
        for i, gx in enumerate(GRID_X):
            for j, gy in enumerate(GRID_Y):
                mask = (
                    (np.abs(pts_body[:, 0] - gx) < SEARCH_RADIUS) &
                    (np.abs(pts_body[:, 1] - gy) < SEARCH_RADIUS)
                )
                if mask.any():
                    # Minimum z matches ray caster firing downward
                    scan[i * GRID_NY + j] = float(pts_body[mask, 2].min()) - robot_base_z

        self._last_scan = scan
        return scan

    def get_imu_data(self):
        """
        Returns (projected_gravity[3], yaw_rate) from ZED 2i IMU.

        projected_gravity: accelerometer reading in body frame (m/s²).
                           When upright and still: approx [0, 0, -9.81].
        yaw_rate:          rotation rate around vertical axis (rad/s).
        """
        self._camera.get_sensors_data(self._sensors_data, sl.TIME_REFERENCE.CURRENT)
        imu = self._sensors_data.get_imu_data()

        accel = imu.get_linear_acceleration()
        gyro  = imu.get_angular_velocity()

        projected_gravity = np.array([accel[0], accel[1], accel[2]], dtype=np.float32)
        yaw_rate = float(gyro[2])

        return projected_gravity, yaw_rate


# ---------------------------------------------------------------------------
# Standalone test — visualise the 7x5 grid live in terminal
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    scanner = TerrainScanner(
        cam_height=0.15,
        cam_forward=0.05,
        tilt_deg=30.0,
    )
    scanner.start()

    print(f"Grid {GRID_NX}x{GRID_NY} | resolution={GRID_RES}m")
    print(f"Forward range : {GRID_X[0]:.2f} – {GRID_X[-1]:.2f} m")
    print(f"Lateral range : {GRID_Y[0]:.2f} – {GRID_Y[-1]:.2f} m")
    print("Ctrl+C to stop\n")

    try:
        while True:
            scan = scanner.get_terrain_scan(robot_base_z=0.0)
            gravity, yaw_rate = scanner.get_imu_data()
            grid = scan.reshape(GRID_NX, GRID_NY)

            print("\033[H\033[J", end="")  # clear terminal
            print("terrain_scan — rows=forward distance, cols=left→right")
            print(f"{'':6s}  {'L':^7s}  {'':^7s}  {'C':^7s}  {'':^7s}  {'R':^7s}")
            for i in range(GRID_NX - 1, -1, -1):  # far row printed first
                vals = "  ".join(f"{grid[i, j]:+.3f}" for j in range(GRID_NY))
                print(f"{GRID_X[i]:.2f}m  {vals}")
            print(f"\ngravity xyz : {gravity[0]:+.3f}  {gravity[1]:+.3f}  {gravity[2]:+.3f}")
            print(f"yaw_rate    : {yaw_rate:+.4f} rad/s")
            print(f"scan range  : {scan.min():+.3f} to {scan.max():+.3f} m")

            time.sleep(0.033)  # ~30 Hz

    except KeyboardInterrupt:
        print("\nStopped.")
        scanner.stop()
