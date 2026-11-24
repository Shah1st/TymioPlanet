"""PLACEHOLDER (filtering owner): fuse wheel odometry with the camera pose."""

from .datatypes import Pose


class PoseFilter:
    """Bayesian filter over the robot pose (x_mm, y_mm, theta_rad).

    The camera pose is absent (None) while the camera is covered or the tag is hidden; the
    estimate must then carry on from odometry alone and recover when the camera returns,
    including after the robot was moved by hand (kidnapping).
    """

    def predict(self, left_speed: float, right_speed: float, dt: float) -> None:
        """Motion step from measured wheel speeds (Thymio units) over dt seconds."""
        raise NotImplementedError

    def update(self, camera_pose: Pose | None) -> Pose:
        """Measurement step; returns the filtered pose."""
        raise NotImplementedError
