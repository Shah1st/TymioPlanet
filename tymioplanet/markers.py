"""Robot pose from the ArUco tag on top of the Thymio.

An ArUco tag is a black square holding a small grid of black and white cells that encodes
an id. The detector finds dark quadrilaterals, rectifies each one, reads its cells and
keeps those matching a dictionary entry. It returns the 4 corners in a fixed order
(top-left, top-right, bottom-right, bottom-left of the upright tag), so one tag gives both
position (the centre) and heading (the direction of its top edge). Template matching
cannot do this: it is neither rotation nor scale invariant (course LAB 2, part 2).
"""

import math
import os

import cv2
import numpy as np

from . import config
from .datatypes import Pose


def detect_tags(frame, dictionary_name=config.ARUCO_DICTIONARY):
    """{tag id: 4x2 corners in image pixels} for every tag found in the frame."""
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    corners, ids, _ = cv2.aruco.ArucoDetector(dictionary, parameters).detectMarkers(grey)
    if ids is None:
        return {}
    return {int(tag_id): np.asarray(quad, dtype=np.float64).reshape(4, 2)
            for tag_id, quad in zip(ids.flatten(), corners)}


def pose_from_tag(corners_px, view, camera_height_mm=config.CAMERA_HEIGHT_MM,
                  tag_height_mm=config.TAG_HEIGHT_MM, tag_offset_mm=config.TAG_OFFSET_MM):
    """Pose of the robot (middle of its wheel axis) in the world frame.

    corners_px: the tag's 4 corners in the image. view: the calibrated FieldView.
    The tag centre and the middle of its top edge are converted to millimetres; the heading
    is the direction from the first to the second.
    """
    corners = np.asarray(corners_px, dtype=np.float64).reshape(4, 2)
    centre_px = corners.mean(axis=0)
    front_px = corners[:2].mean(axis=0)
    points = view.to_world([centre_px, front_px])
    if camera_height_mm:
        # The point under an overhead camera is the one seen at the image centre.
        width_px, height_px = view.image_size
        nadir = view.to_world([[width_px / 2, height_px / 2]])[0]
        points = correct_parallax(points, nadir, camera_height_mm, tag_height_mm)
    centre, front = points
    theta = math.atan2(front[1] - centre[1], front[0] - centre[0])
    # Step back from the tag centre to the middle of the wheel axis.
    offset_x, offset_y = tag_offset_mm
    x = centre[0] - (offset_x * math.cos(theta) - offset_y * math.sin(theta))
    y = centre[1] - (offset_x * math.sin(theta) + offset_y * math.cos(theta))
    return Pose(float(x), float(y), theta)


def save_tags(folder="tags", side_px=600):
    """Writes the five tags as pictures to print, each with its white border and a label."""
    os.makedirs(folder, exist_ok=True)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config.ARUCO_DICTIONARY))
    places = ["bottom-left (origin)", "bottom-right", "top-right", "top-left"]
    labels = {config.ROBOT_TAG_ID: ("robot", "ROBOT: this edge is the front")}
    for tag_id, place in zip(config.CORNER_TAG_IDS, places):
        labels[tag_id] = (f"corner_{tag_id}", f"CORNER {tag_id}: {place}")
    paths = []
    for tag_id, (name, label) in labels.items():
        border = side_px // 6
        tag = cv2.aruco.generateImageMarker(dictionary, tag_id, side_px)
        page = cv2.copyMakeBorder(tag, border + 60, border, border, border,
                                  cv2.BORDER_CONSTANT, value=255)
        cv2.putText(page, f"id {tag_id}  {label}", (border, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2, cv2.LINE_AA)
        paths.append(os.path.join(folder, f"tag_{tag_id}_{name}.png"))
        cv2.imwrite(paths[-1], page)
    return paths


def correct_parallax(points_mm, nadir_mm, camera_height_mm, tag_height_mm):
    """Moves points seen at tag height back to where they stand on the floor.

    A point at height h, seen from a camera at height H, appears farther from the point
    under the camera (the nadir) by the factor H / (H - h). This undoes it.
    """
    points = np.asarray(points_mm, dtype=np.float64).reshape(-1, 2)
    nadir = np.asarray(nadir_mm, dtype=np.float64).reshape(1, 2)
    return nadir + (points - nadir) * (camera_height_mm - tag_height_mm) / camera_height_mm


if __name__ == "__main__":
    for saved in save_tags():
        print("saved", saved)
