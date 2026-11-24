"""Motion control: the path follower. The Thymio link is still a placeholder."""

import math
from dataclasses import dataclass

from . import config
from .datatypes import Pose


@dataclass
class FollowerParams:
    wheelbase_mm: float = config.WHEELBASE_MM  # distance between the wheels (measure yours)
    units_per_mm_s: float = config.THYMIO_UNITS_PER_MM_S  # motor units per mm/s (calibrate!)
    v_max_mm_s: float = 100.0         # cruising speed
    v_min_mm_s: float = 25.0          # don't crawl slower than this (motors stall)
    k_heading: float = 2.5            # rad/s of turn rate per rad of heading error
    omega_max: float = 3.0            # rad/s turn-rate limit
    waypoint_tol_mm: float = 30.0     # intermediate waypoint reached
    goal_tol_mm: float = 15.0         # final goal reached
    slow_radius_mm: float = 100.0     # start slowing down this far from the goal
    turn_in_place_rad: float = 1.0    # above this heading error, rotate without driving
    max_motor: int = 500              # Thymio motor limit


def wrap_angle(a: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def _deadband(w, p):
    # a wheel commanded below v_min (but not zero) is pushed up to v_min in its direction
    return 0.0 if w == 0 else math.copysign(max(abs(w), p.v_min_mm_s), w)


def follow_path_step(pose, path, idx, p: FollowerParams | None = None):
    """One control step of waypoint following.

    pose : Pose, or (x, y, theta), current estimate in mm and rad
    path : list of (x, y) waypoints in mm, in order
    idx  : index of the waypoint currently being tracked (start with 0)

    Returns (left, right, idx, done):
        left, right : motor targets (ints, Thymio units) to send to the robot
        idx         : updated waypoint index (keep it between calls)
        done        : True once the final goal is reached (motors = 0)
    """
    p = p or FollowerParams()
    if isinstance(pose, Pose):
        pose = (pose.x_mm, pose.y_mm, pose.theta_rad)
    x, y, theta = pose
    last = len(path) - 1

    # Advance past waypoints that are already reached (or skipped).
    while idx < last:
        wx, wy = path[idx]
        if math.hypot(wx - x, wy - y) < p.waypoint_tol_mm:
            idx += 1
        else:
            break

    gx, gy = path[idx]
    dist = math.hypot(gx - x, gy - y)

    if idx == last and dist < p.goal_tol_mm:
        return 0, 0, idx, True

    # Heading error toward the current waypoint.
    err = wrap_angle(math.atan2(gy - y, gx - x) - theta)

    # Forward speed: reduce with heading error, and ramp down near the goal.
    if abs(err) > p.turn_in_place_rad:
        v = 0.0
    else:
        v = p.v_max_mm_s * math.cos(err)
        if idx == last:
            v = min(v, p.v_max_mm_s * dist / p.slow_radius_mm)
        v = max(v, p.v_min_mm_s) if v > 0 else 0.0

    # Rotational speed: if in place, needs enough wheel speed to overcome friction.
    omega = max(-p.omega_max, min(p.omega_max, p.k_heading * err))
    if v == 0.0 and abs(omega) * p.wheelbase_mm / 2 < p.v_min_mm_s:
        omega = math.copysign(p.v_min_mm_s * 2 / p.wheelbase_mm, err)

    # Unicycle model -> wheel speeds (mm/s) -> motor units.
    v_l = _deadband(v - omega * p.wheelbase_mm / 2, p)
    v_r = _deadband(v + omega * p.wheelbase_mm / 2, p)
    left = v_l * p.units_per_mm_s
    right = v_r * p.units_per_mm_s

    # Saturate while preserving the turn ratio.
    m = max(abs(left), abs(right))
    if m > p.max_motor:
        left *= p.max_motor / m
        right *= p.max_motor / m

    return int(round(left)), int(round(right)), idx, False


class Thymio:
    """PLACEHOLDER: connection to the robot (tdmclient, Thymio Device Manager running)."""

    def connect(self) -> None:
        raise NotImplementedError

    def read_sensors(self) -> dict:
        """{'prox_horizontal': [7 ints], 'left_speed': int, 'right_speed': int}, Thymio units."""
        raise NotImplementedError

    def set_motors(self, left: int, right: int) -> None:
        """Wheel targets in Thymio units (-500..500)."""
        raise NotImplementedError
