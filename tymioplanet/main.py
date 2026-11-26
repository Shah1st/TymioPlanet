"""Runs the whole project: vision, pose filter and route planning. The robot link is missing.

    python -m tymioplanet.main --source synthetic     no hardware needed
    python -m tymioplanet.main --source 0             webcam number 0

The map is made once, from the first frame showing the four corner tags.
Keys in the window: q quit, m retake the map.
Synthetic source only: h cover / uncover the camera.
"""

import argparse

import cv2

from . import camera, control, filtering, local_avoidance, planning, visualize
from .vision import VisionSystem

_WINDOW = "TymioPlanet"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default="synthetic",
                        help="'synthetic', a camera index, or a picture / video path")
    parser.add_argument("--frames", type=int, default=0, help="stop after this many frames")
    parser.add_argument("--no-view", action="store_true", help="print instead of opening a window")
    args = parser.parse_args()

    source = camera.open_source(args.source)
    vision_system = VisionSystem()
    pose_filter = filtering.PoseFilter()
    path = None
    count = 0
    try:
        while True:
            frame = source.read()
            observation = vision_system.update(frame)

            pose = pose_filter.update(observation.pose) or observation.pose
            if path is None and observation.world_map is not None and pose is not None:
                path = _plan(observation.world_map, pose)
            # Next, once control.Thymio exists: send local_avoidance.THYMIO_PROGRAM once, then
            # at every step read the wheel speeds, pose_filter.predict(...), and write the
            # output of control.follow_path_step(pose, path, idx) to following_left_speed /
            # following_right_speed on the robot.

            count += 1
            if args.no_view:
                print(count, observation.camera_ok, observation.pose)
            elif frame is not None:
                cv2.imshow(_WINDOW, visualize.render_view(frame, vision_system, observation, path))
                key = cv2.waitKey(30) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("m") and vision_system.snap(frame) is not None:
                    path = None  # new map: plan again
                if key == ord("h") and isinstance(source, camera.SyntheticCamera):
                    source.covered = not source.covered
            if args.frames and count >= args.frames:
                break
    finally:
        source.release()
        cv2.destroyAllWindows()


def _plan(world_map, pose):
    """The route, or an empty list (so it is not tried again every frame) when there is none."""
    try:
        return planning.plan_route(world_map, pose)
    except RuntimeError as error:
        print("no route:", error)
        return []


if __name__ == "__main__":
    main()
