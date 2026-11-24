"""Runs the whole project. Today that is the vision part plus placeholders for the rest.

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

            # --- teammates' modules; each is skipped until it is implemented ---
            pose = _optional(pose_filter.update, observation.pose) or observation.pose
            if path is None and observation.world_map is not None and pose is not None:
                path = _optional(planning.plan_route, observation.world_map, pose)
            # Next, once control.Thymio exists: read sensors, pose_filter.predict(...),
            # local_avoidance.avoid(prox) or control.follow_path_step(pose, path, idx), set motors.

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


def _optional(function, *arguments):
    """Result of a teammate's function, or None while it is still a placeholder."""
    try:
        return function(*arguments)
    except NotImplementedError:
        return None


if __name__ == "__main__":
    main()
