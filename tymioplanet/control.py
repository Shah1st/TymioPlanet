"""Motion control: the path follower and the link to the robot."""

import math
import time
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
    """Connection to the robot through the Thymio Device Manager (tdmclient).

    Thymio Suite, or its device manager alone, must be running and the robot switched on.

        with Thymio() as robot:
            sensors = robot.read_sensors()
            robot.set_motors(50, 50)
    """

    _VARIABLES = {"prox.horizontal", "motor.left.speed", "motor.right.speed"}

    def __init__(self):
        self._client = None
        self._node = None
        self._aw = None

    def connect(self, timeout_s: float = 10.0) -> None:
        """Finds the robot, takes control of it and stops the program stored on it.

        That program has to be stopped: it can write the motor targets at every tick and
        cancel the ones sent from here.
        """
        import tdmclient  # imported here so the rest of the project runs without it

        try:
            self._client = tdmclient.ClientAsync()
        except ConnectionRefusedError as error:
            raise ConnectionError(
                "Thymio Device Manager is not running: open Thymio Suite") from error
        self._aw = tdmclient.aw
        deadline = time.monotonic() + timeout_s
        while not self._client.nodes:
            if time.monotonic() > deadline:
                self.close()
                raise ConnectionError("no Thymio found: is it switched on and paired?")
            self._client.process_waiting_messages()
            time.sleep(0.05)
        self._node = self._client.nodes[0]
        self._aw(self._node.lock())
        self._aw(self._node.stop())
        self._aw(self._node.wait_for_variables(self._VARIABLES))

    def read_sensors(self) -> dict:
        """{'prox_horizontal': [7 ints], 'left_speed': int, 'right_speed': int}, Thymio units.

        Proximity: 0 when nothing is seen, up to about 4500 for an object touching the robot.
        Speeds are the measured wheel speeds, not the targets.
        """
        self._client.process_waiting_messages()  # take in the latest values from the robot
        values = self._node.v
        return {"prox_horizontal": list(values.prox.horizontal),
                "left_speed": int(values.motor.left.speed),
                "right_speed": int(values.motor.right.speed)}

    def set_motors(self, left: int, right: int) -> None:
        """Wheel targets in Thymio units (-500..500)."""
        # Sent without waiting for the robot's acknowledgement: tdmclient checks for it only
        # every 0.1 s, which would hold up every control step by that much.
        self._node.send_set_variables(self._targets(left, right))

    @staticmethod
    def _targets(left, right):
        left, right = (max(-500, min(500, int(round(value)))) for value in (left, right))
        return {"motor.left.target": [left], "motor.right.target": [right]}

    def close(self) -> None:
        """Stops the wheels and gives the robot back."""
        if self._node is not None:
            try:
                self._aw(self._node.set_variables(self._targets(0, 0)))  # wait: it must arrive
                self._aw(self._node.unlock())
            finally:
                self._node = None
        if self._client is not None:
            self._client.disconnect()
            self._client = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exception):
        self.close()


if __name__ == "__main__":
    # python -m tymioplanet.control : checks the connection. The robot does not move.
    with Thymio() as thymio:
        for _ in range(3):
            time.sleep(0.3)
            print(thymio.read_sensors())
    print("robot link ok")
