"""Pose filtering: an extended Kalman filter fusing wheel odometry with the camera pose.

PoseEKF is the filter itself. PoseFilter wraps it in the interface the main loop uses:
Thymio speed units in, a Pose out, and the camera pose may be None (covered camera).
"""

import math

import numpy as np

from . import config
from .control import wrap_angle
from .datatypes import Pose


class PoseEKF:
    def __init__(self, x0, P0, wheelbase_mm=config.WHEELBASE_MM,
                 sigma_v_rel=0.05, sigma_v_abs=3.0, history_s=1.0):
        """sigma_v_rel / sigma_v_abs: wheel-speed noise std = rel*|v| + abs (mm/s).
        These must also cover systematic odometry error (scale, slip)."""
        self.L = wheelbase_mm
        self.sv_rel, self.sv_abs = sigma_v_rel, sigma_v_abs
        self.history_s = history_s
        self.x = np.array(x0, float)
        self.P = np.array(P0, float)
        self.t = 0.0
        self.hist = [(0.0, self.x.copy(), self.P.copy(), 0.0, 0.0, 0.0)]
        self.last_nis = None
        self.rejected = 0

    # ---- core math -------------------------------------------------------
    def _propagate(self, x, P, vl, vr, dt):
        L = self.L
        v, w = 0.5 * (vl + vr), (vr - vl) / L
        th = x[2]
        thm = th + 0.5 * w * dt
        c, s = math.cos(thm), math.sin(thm)
        xn = np.array([x[0] + v * dt * c,
                       x[1] + v * dt * s,
                       wrap_angle(th + w * dt)])
        # Jacobian wrt state
        F = np.array([[1, 0, -v * dt * s],
                      [0, 1,  v * dt * c],
                      [0, 0, 1]])
        # Jacobian wrt (vl, vr) to map wheel noise into state noise
        k = dt * dt / (2 * L)
        G = np.array([[0.5 * dt * c + v * s * k, 0.5 * dt * c - v * s * k],
                      [0.5 * dt * s - v * c * k, 0.5 * dt * s + v * c * k],
                      [-dt / L,                  dt / L]])
        sl = self.sv_rel * abs(vl) + self.sv_abs
        sr = self.sv_rel * abs(vr) + self.sv_abs
        Q = G @ np.diag([sl * sl, sr * sr]) @ G.T
        return xn, F @ P @ F.T + Q

    def _correct(self, x, P, z, R, gate):
        y = np.asarray(z, float) - x
        y[2] = wrap_angle(y[2])
        S = P + R                       # H = I
        nis = float(y @ np.linalg.solve(S, y))
        if gate is not None and nis > gate:
            return x, P, nis, False
        K = P @ np.linalg.inv(S)
        xn = x + K @ y
        xn[2] = wrap_angle(xn[2])
        I = np.eye(3)
        Pn = (I - K) @ P @ (I - K).T + K @ R @ K.T      # Joseph form
        return xn, Pn, nis, True

    # ---- main-interface --------------------------------------------------

    def predict(self, vl, vr, dt):
        """vl, vr: measured wheel speeds in mm/s over the last dt seconds."""
        self.x, self.P = self._propagate(self.x, self.P, vl, vr, dt)
        self.t += dt
        self.hist.append((self.t, self.x.copy(), self.P.copy(), vl, vr, dt))
        while len(self.hist) > 2 and self.hist[1][0] < self.t - self.history_s:
            self.hist.pop(0)

    def update(self, z, R, t_meas=None, gate=11.34):
        """Camera pose z=[x,y,theta] taken at time t_meas (filter clock).
        gate: chi-square threshold on the normalized innovation (3 dof,
        11.34 = 99%). Returns True if the measurement was accepted."""
        if t_meas is None or t_meas >= self.t:
            x, P, nis, ok = self._correct(self.x, self.P, z, R, gate)
            self.last_nis = nis
            if ok:
                self.x, self.P = x, P
                self.hist[-1] = (self.t, x.copy(), P.copy(), *self.hist[-1][3:])
            else:
                self.rejected += 1
            return ok

        # Delayed measurement: roll back to nearest snapshot, correct, replay.
        k = min(range(len(self.hist)), key=lambda i: abs(self.hist[i][0] - t_meas))
        if t_meas < self.hist[0][0] - 1e-9:
            return False                 # older than the history buffer
        t, x, P, vl, vr, dt = self.hist[k]
        x, P, nis, ok = self._correct(x, P, z, R, gate)
        self.last_nis = nis
        if not ok:
            self.rejected += 1
            return False
        self.hist[k] = (t, x.copy(), P.copy(), vl, vr, dt)
        for j in range(k + 1, len(self.hist)):
            tj, _, _, vlj, vrj, dtj = self.hist[j]
            x, P = self._propagate(x, P, vlj, vrj, dtj)
            self.hist[j] = (tj, x.copy(), P.copy(), vlj, vrj, dtj)
        self.x, self.P = x.copy(), P.copy()
        return True


class PoseFilter:
    """PoseEKF behind the interface of the main loop.

    The filter starts at the first camera pose. While the camera is covered (pose None) the
    estimate carries on from odometry. The EKF rejects camera poses far from its estimate
    (outliers); if that keeps happening, the estimate is the one that is wrong (robot moved
    by hand, or odometry drifted during a long cover), so it restarts from the camera.
    """

    def __init__(self, camera_std=config.CAMERA_POSE_STD,
                 units_per_mm_s=config.THYMIO_UNITS_PER_MM_S,
                 reset_after=config.FILTER_RESET_AFTER_REJECTIONS):
        self.R = np.diag(np.square(camera_std))
        self.units_per_mm_s = units_per_mm_s
        self.reset_after = reset_after
        self.ekf = None
        self._rejections_in_a_row = 0

    def predict(self, left_speed: float, right_speed: float, dt: float) -> None:
        """Motion step from measured wheel speeds (Thymio units) over dt seconds."""
        if self.ekf is not None:
            self.ekf.predict(left_speed / self.units_per_mm_s,
                             right_speed / self.units_per_mm_s, dt)

    def update(self, camera_pose: Pose | None) -> Pose | None:
        """Measurement step; returns the filtered pose (None before the first camera pose)."""
        if camera_pose is not None:
            z = [camera_pose.x_mm, camera_pose.y_mm, camera_pose.theta_rad]
            if self.ekf is None:
                self.ekf = PoseEKF(z, self.R)
            elif self.ekf.update(z, self.R):
                self._rejections_in_a_row = 0
            else:
                self._rejections_in_a_row += 1
                if self._rejections_in_a_row >= self.reset_after:
                    self.ekf = PoseEKF(z, self.R)
                    self._rejections_in_a_row = 0
        if self.ekf is None:
            return None
        return Pose(*map(float, self.ekf.x))

    @property
    def covariance(self):
        """3x3 covariance of (x mm, y mm, theta rad), or None before the first camera pose."""
        return None if self.ekf is None else self.ekf.P
