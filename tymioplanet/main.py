"""Runs the whole project: vision, pose filter, route planning and driving the route.

    python -m tymioplanet.main --source synthetic     no hardware needed
    python -m tymioplanet.main --source 0             webcam number 0, robot not moved
    python -m tymioplanet.main --source 0 --robot     the same, and the Thymio drives the route

The map is made once, from the first frame showing the four corner tags.
Keys in the window: q quit, m retake the map.
Synthetic source only: h cover / uncover the camera.
"""

import argparse
import time

import cv2

from . import camera, control, filtering, local_avoidance, planning, visualize
from .vision import VisionSystem

_WINDOW = "TymioPlanet"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default="synthetic",
                        help="'synthetic', a camera index, or a picture / video path")
    parser.add_argument("--frames", type=int, default=0, help="stop after this many frames")
    parser.add_argument("--robot", action="store_true",
                        help="connect to the Thymio and drive the route (needs a real camera)")
    parser.add_argument("--no-view", action="store_true", help="print instead of opening a window")
    args = parser.parse_args()
    if args.robot and args.source == "synthetic":
        parser.error("--robot needs a real camera: the synthetic robot never moves")

    source = camera.open_source(args.source)
    vision_system = VisionSystem()
    pose_filter = filtering.PoseFilter()
    # A webcam already paces the loop (read waits for the next frame); the others need a pause.
    key_wait_ms = 1 if isinstance(source, camera.LiveCamera) else 30
    path = None
    index = 0
    count = 0
    robot = None
    last_time = time.monotonic()
    try:
        if args.robot:
            robot = control.Thymio()
            robot.connect()
        while True:
            frame = source.read()
            observation = vision_system.update(frame)

            now = time.monotonic()
            if robot is not None:  # odometry: the only pose source while the camera is covered
                sensors = robot.read_sensors()
                pose_filter.predict(sensors["left_speed"], sensors["right_speed"], now - last_time)
            last_time = now
            pose = pose_filter.update(observation.pose) or observation.pose
            if path is None and observation.world_map is not None and pose is not None:
                path = _plan(observation.world_map, pose)
                index = 0
            if robot is not None:
                index = drive(robot, pose, path, index)
            # Local avoidance is not connected yet: local_avoidance.THYMIO_PROGRAM would be
            # sent to the robot once, and the speeds above written to its
            # following_left_speed / following_right_speed instead of the motor targets.

            count += 1
            if args.no_view:
                print(count, observation.camera_ok, observation.pose)
            elif frame is not None:
                cv2.imshow(_WINDOW, visualize.render_view(frame, vision_system, observation, path))
                key = cv2.waitKey(key_wait_ms) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("m") and vision_system.snap(frame) is not None:
                    path = None  # new map: plan again
                if key == ord("h") and isinstance(source, camera.SyntheticCamera):
                    source.covered = not source.covered
            if args.frames and count >= args.frames:
                break
    finally:
        if robot is not None:
            robot.close()  # also stops the wheels
        source.release()
        cv2.destroyAllWindows()


def drive(robot, pose, path, index):
    """One control step: wheel speeds towards the current waypoint. Returns the new index.

    The robot stands still while there is no pose or no route.
    """
    if pose is None or not path:
        robot.set_motors(0, 0)
        return index
    left, right, index, _ = control.follow_path_step(pose, path, index)
    robot.set_motors(left, right)  # both 0 once the last waypoint is reached
    return index


def _plan(world_map, pose):
    """The route, or an empty list (so it is not tried again every frame) when there is none."""
    try:
        return planning.plan_route(world_map, pose)
    except RuntimeError as error:
        print("no route:", error)
        return []


if __name__ == "__main__":
    main()
