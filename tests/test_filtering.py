import math

import numpy as np

from tymioplanet import config
from tymioplanet.control import wrap_angle
from tymioplanet.datatypes import Pose
from tymioplanet.filtering import PoseEKF, PoseFilter

_UNITS = config.THYMIO_UNITS_PER_MM_S
_DT = 0.1


def test_predict_matches_a_numerical_jacobian():
    ekf = PoseEKF([0, 0, 0], np.eye(3))
    x0, vl, vr, h = np.array([100.0, 200.0, 0.3]), 80.0, 120.0, 1e-5
    xn, P = ekf._propagate(x0, np.zeros((3, 3)), vl, vr, _DT)
    G = np.column_stack([(ekf._propagate(x0, np.zeros((3, 3)), vl + h, vr, _DT)[0] - xn) / h,
                         (ekf._propagate(x0, np.zeros((3, 3)), vl, vr + h, _DT)[0] - xn) / h])
    sl, sr = 0.05 * vl + 3.0, 0.05 * vr + 3.0
    assert np.allclose(P, G @ np.diag([sl ** 2, sr ** 2]) @ G.T, atol=1e-6)


def _run(steps, camera_visible, kidnap_at=None, seed=0):
    """Robot on a gentle arc; odometry has a few percent of wheel-speed error."""
    rng = np.random.default_rng(seed)
    pose_filter = PoseFilter()
    true = np.array([200.0, 200.0, 0.0])
    errors = []
    for k in range(steps):
        vl, vr = 80.0, 90.0  # mm/s
        v, omega = (vl + vr) / 2, (vr - vl) / config.WHEELBASE_MM
        true += [v * math.cos(true[2]) * _DT, v * math.sin(true[2]) * _DT, omega * _DT]
        true[2] = wrap_angle(true[2])
        if k == kidnap_at:
            true = np.array([800.0, 600.0, 2.0])
        pose_filter.predict(vl * 1.03 * _UNITS, vr * 0.98 * _UNITS, _DT)
        measured = None
        if camera_visible(k):
            measured = Pose(*(true + rng.normal(0, config.CAMERA_POSE_STD)))
        estimate = pose_filter.update(measured)
        errors.append(math.hypot(estimate.x_mm - true[0], estimate.y_mm - true[1])
                      if estimate else None)
    return errors


def test_no_estimate_before_the_first_camera_pose():
    pose_filter = PoseFilter()
    pose_filter.predict(100, 100, _DT)
    assert pose_filter.update(None) is None and pose_filter.covariance is None
    assert pose_filter.update(Pose(10, 20, 0.5)) == Pose(10, 20, 0.5)


def test_tracks_with_the_camera():
    errors = _run(300, lambda k: True)
    assert max(errors[20:]) < 10.0


def test_carries_on_through_a_short_cover_and_recovers_after_a_long_one():
    errors = _run(500, lambda k: not 100 <= k < 300)  # covered for 20 s
    assert errors[110] < 20.0  # 1 s into the cover: odometry holds
    assert errors[299] > 50.0  # 20 s of biased odometry has drifted
    assert max(errors[320:]) < 10.0  # camera back: the estimate returns to it


def test_recovers_from_kidnapping():
    errors = _run(300, lambda k: True, kidnap_at=150)
    assert errors[150] > 300.0
    assert max(errors[160:]) < 10.0
