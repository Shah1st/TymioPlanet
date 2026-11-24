"""Renders camera frames of a made-up scene, so the vision code can be tested without hardware.

The scene is drawn flat in world coordinates, then projected into a camera image through a
homography, which is exactly what a real camera does to a flat sheet.
"""

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import config
from .datatypes import Pose

# BGR colours
RED = (40, 40, 220)
GREEN = (60, 180, 60)
BLUE = (220, 90, 30)
BLACK = (30, 30, 30)
SHEET_WHITE = (245, 245, 245)
TABLE_GREY = (70, 70, 70)
ROBOT_BODY = (232, 232, 232)

_DRAW_SCALE = 2.0  # pixels per mm of the flat drawing, finer than the camera sees
TAG_SIDE_MM = 60.0
ROBOT_SIDE_MM = 110.0


@dataclass(frozen=True)
class Shape:
    kind: str  # "circle" or "square"
    x_mm: float
    y_mm: float
    size_mm: float  # radius of a circle, side of a square
    color: tuple


@dataclass
class Scene:
    shapes: list[Shape] = field(default_factory=list)
    robot: Pose | None = None
    width_mm: float = config.SHEET_WIDTH_MM
    height_mm: float = config.SHEET_HEIGHT_MM


def default_sheet_corners(image_size=(1280, 720)):
    """Image corners of the sheet for a camera that is almost, but not exactly, overhead.

    Order: world (0, 0), (W, 0), (W, H), (0, H).
    """
    w, h = image_size
    return np.array(
        [[0.07 * w, 0.93 * h], [0.94 * w, 0.90 * h], [0.91 * w, 0.06 * h], [0.10 * w, 0.09 * h]],
        dtype=np.float32,
    )


def demo_scene():
    """Earth, three planets, asteroids with a gap too narrow for the robot, and a black box."""
    shapes = [
        Shape("circle", 150, 150, 55, BLUE),
        Shape("circle", 1000, 200, 50, GREEN),
        Shape("circle", 950, 680, 50, GREEN),
        Shape("circle", 300, 650, 50, GREEN),
        Shape("circle", 550, 250, 60, RED),
        Shape("circle", 550, 430, 60, RED),  # 60 mm gap to the one above: a trap
        Shape("circle", 780, 480, 55, RED),
        Shape("square", 330, 400, 90, BLACK),
    ]
    return Scene(shapes=shapes, robot=Pose(340, 170, math.radians(20)))


def render(scene, image_size=(1280, 720), sheet_corners=None, noise_std=0.0, seed=0):
    """Returns the BGR camera frame of the scene."""
    if sheet_corners is None:
        sheet_corners = default_sheet_corners(image_size)
    flat = _draw_flat(scene)
    w, h = scene.width_mm * _DRAW_SCALE, scene.height_mm * _DRAW_SCALE
    # flat-drawing corners of world (0,0), (W,0), (W,H), (0,H); the drawing has y pointing down
    flat_corners = np.array([[0, h], [w, h], [w, 0], [0, 0]], dtype=np.float32)
    to_image = cv2.getPerspectiveTransform(flat_corners, np.asarray(sheet_corners, np.float32))
    frame = cv2.warpPerspective(
        flat, to_image, image_size, flags=cv2.INTER_AREA,
        borderMode=cv2.BORDER_CONSTANT, borderValue=TABLE_GREY,
    )
    if noise_std > 0:
        noise = np.random.default_rng(seed).normal(0.0, noise_std, frame.shape)
        frame = np.clip(frame.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    return frame


def _to_flat(scene, x_mm, y_mm):
    return x_mm * _DRAW_SCALE, (scene.height_mm - y_mm) * _DRAW_SCALE


def _draw_flat(scene):
    size = (round(scene.width_mm * _DRAW_SCALE), round(scene.height_mm * _DRAW_SCALE))
    flat = np.full((size[1], size[0], 3), SHEET_WHITE, dtype=np.uint8)
    for shape in scene.shapes:
        u, v = _to_flat(scene, shape.x_mm, shape.y_mm)
        if shape.kind == "circle":
            cv2.circle(flat, (round(u), round(v)), round(shape.size_mm * _DRAW_SCALE),
                       shape.color, -1, cv2.LINE_AA)
        else:
            half = shape.size_mm * _DRAW_SCALE / 2
            cv2.rectangle(flat, (round(u - half), round(v - half)),
                          (round(u + half), round(v + half)), shape.color, -1)
    if scene.robot is not None:
        _draw_robot(flat, scene)
    return flat


def _draw_robot(flat, scene):
    """A light square body with the ArUco tag on top, tag's top edge towards the front."""
    pose = scene.robot
    u, v = _to_flat(scene, pose.x_mm, pose.y_mm)
    # The tag image has its top edge up, which is heading 90 degrees; rotate by the difference.
    angle_deg = math.degrees(pose.theta_rad) - 90.0

    def paste(patch):
        side = patch.shape[0]
        matrix = cv2.getRotationMatrix2D((side / 2, side / 2), angle_deg, 1.0)
        matrix[:, 2] += (u - side / 2, v - side / 2)
        size = (flat.shape[1], flat.shape[0])
        layer = cv2.warpAffine(patch, matrix, size, flags=cv2.INTER_LINEAR)
        mask = cv2.warpAffine(np.full(patch.shape[:2], 255, np.uint8), matrix, size)
        flat[mask > 127] = layer[mask > 127]

    body_px = round(ROBOT_SIDE_MM * _DRAW_SCALE)
    paste(np.full((body_px, body_px, 3), ROBOT_BODY, dtype=np.uint8))

    tag_px = round(TAG_SIDE_MM * _DRAW_SCALE)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config.ARUCO_DICTIONARY))
    tag = cv2.aruco.generateImageMarker(dictionary, config.ROBOT_TAG_ID, tag_px)
    border = tag_px // 6  # white quiet zone the detector needs around the black frame
    tag = cv2.copyMakeBorder(tag, border, border, border, border, cv2.BORDER_CONSTANT, value=255)
    paste(cv2.cvtColor(tag, cv2.COLOR_GRAY2BGR))
