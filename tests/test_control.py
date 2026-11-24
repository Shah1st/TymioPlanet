import math

from tymioplanet.control import FollowerParams, follow_path_step, wrap_angle
from tymioplanet.datatypes import Pose


def _drive(path, pose, params=None, seconds=120.0, dt=0.05):
    """Kinematic simulation of a differential-drive robot under the follower."""
    p = params or FollowerParams()
    x, y, theta = pose
    idx, t, done = 0, 0.0, False
    while t < seconds:
        left, right, idx, done = follow_path_step(Pose(x, y, theta), path, idx, p)
        assert abs(left) <= p.max_motor and abs(right) <= p.max_motor
        if done:
            break
        v_l, v_r = left / p.units_per_mm_s, right / p.units_per_mm_s
        v, omega = (v_l + v_r) / 2, (v_r - v_l) / p.wheelbase_mm
        x += v * math.cos(theta) * dt
        y += v * math.sin(theta) * dt
        theta = wrap_angle(theta + omega * dt)
        t += dt
    return done, (x, y)


def test_reaches_the_goal_of_a_path_with_corners():
    path = [(0, 0), (300, 0), (300, 300), (600, 300), (600, 0)]
    done, (x, y) = _drive(path, (-20.0, 40.0, math.radians(90)))
    assert done and math.hypot(x - 600, y) < FollowerParams().goal_tol_mm


def test_follows_a_dense_path_as_a_grid_planner_gives():
    path = [(x, 0) for x in range(0, 301, 10)] + [(300, y) for y in range(10, 301, 10)]
    done, (x, y) = _drive(path, (0.0, 0.0, math.pi))  # starts facing the wrong way
    assert done and math.hypot(x - 300, y - 300) < FollowerParams().goal_tol_mm


def test_positive_heading_error_turns_left():
    # Target straight ahead-left: the right wheel must run faster (counter-clockwise turn).
    left, right, _, _ = follow_path_step((0, 0, 0), [(0, 0), (300, 100)], 1)
    assert right > left


def test_stops_at_the_goal():
    assert follow_path_step((299, 0, 0), [(0, 0), (300, 0)], 1) == (0, 0, 1, True)
