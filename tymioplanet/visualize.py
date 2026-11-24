"""Live view: camera image, rectified top-down image and occupancy grid, side by side.

Status text goes in a strip under the panels, not on top of the images (course "Common
pitfalls": do not overlay the parameters on the image itself).
"""

import math

import cv2
import numpy as np

from .datatypes import OCCUPIED

_PANEL_HEIGHT = 360
_STRIP_HEIGHT = 32
_GREEN, _BLUE, _RED, _ORANGE = (60, 180, 60), (220, 90, 30), (40, 40, 220), (0, 140, 255)


def render_view(frame, vision_system, observation, path_mm=None):
    """One BGR image with the three panels and a status strip.

    path_mm: optional list of (x, y) world points from the planner, drawn on the grid.
    """
    panels = [_camera_panel(frame, vision_system.view)]
    if vision_system.view is not None and observation.world_map is not None:
        panels.append(_top_down_panel(vision_system, observation))
        panels.append(_grid_panel(vision_system.view, observation, path_mm))
    panels = [_fit_height(panel) for panel in panels]
    row = np.hstack(panels)
    strip = np.full((_STRIP_HEIGHT, row.shape[1], 3), 30, dtype=np.uint8)
    cv2.putText(strip, _status(observation), (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (230, 230, 230), 1, cv2.LINE_AA)
    return np.vstack([row, strip])


def _status(observation):
    if not observation.camera_ok:
        return "camera: COVERED    pose: none (odometry only)    map: last known"
    pose = observation.pose
    pose_text = "tag not visible" if pose is None else (
        f"x={pose.x_mm:.0f} mm  y={pose.y_mm:.0f} mm  theta={math.degrees(pose.theta_rad):.0f} deg")
    world_map = observation.world_map
    return (f"camera: ok    pose: {pose_text}    obstacles: {len(world_map.obstacles)}"
            f"  planets: {len(world_map.waypoints)}  earth: {'yes' if world_map.home else 'no'}"
            f"{'    MAP CHANGED' if observation.map_changed else ''}")


def _camera_panel(frame, view):
    panel = frame.copy()
    if view is not None:
        cv2.polylines(panel, [np.round(view.corners_px).astype(np.int32)], True, _ORANGE, 2)
    return panel


def _top_down_panel(vision_system, observation):
    view = vision_system.view
    panel = vision_system.top_down.copy()
    world_map = observation.world_map
    blobs = world_map.obstacles + world_map.waypoints + ([world_map.home] if world_map.home else [])
    colours = {"obstacle": _RED, "waypoint": _GREEN, "home": _BLUE}
    for blob in blobs:
        centre = view.world_to_top_down([[blob.x_mm, blob.y_mm]])[0]
        cv2.circle(panel, _point(centre), round(blob.radius_mm / view.mm_per_pixel) + 3,
                   colours[blob.kind], 2, cv2.LINE_AA)
    if observation.pose is not None:
        _draw_pose(panel, observation.pose, lambda x, y: view.world_to_top_down([[x, y]])[0])
    return panel


def _grid_panel(view, observation, path_mm):
    world_map = observation.world_map
    cell_px = 4
    image = np.full((*world_map.grid.shape, 3), 255, dtype=np.uint8)
    image[world_map.grid_inflated == OCCUPIED] = (170, 170, 170)  # grown by the robot radius
    image[world_map.grid == OCCUPIED] = (0, 0, 0)  # what the camera saw
    panel = cv2.resize(np.flipud(image), None, fx=cell_px, fy=cell_px,
                       interpolation=cv2.INTER_NEAREST)
    scale = cell_px / world_map.cell_size_mm

    def to_panel(x_mm, y_mm):
        return x_mm * scale, panel.shape[0] - y_mm * scale

    for blob in world_map.waypoints:
        cv2.circle(panel, _point(to_panel(blob.x_mm, blob.y_mm)), 6, _GREEN, -1, cv2.LINE_AA)
    if world_map.home is not None:
        cv2.circle(panel, _point(to_panel(world_map.home.x_mm, world_map.home.y_mm)), 7, _BLUE,
                   -1, cv2.LINE_AA)
    if path_mm:
        points = np.array([_point(to_panel(x, y)) for x, y in path_mm], dtype=np.int32)
        cv2.polylines(panel, [points], False, _ORANGE, 2, cv2.LINE_AA)
    if observation.pose is not None:
        _draw_pose(panel, observation.pose, to_panel)
    return panel


def _draw_pose(panel, pose, to_pixels, length_mm=120.0):
    tail = to_pixels(pose.x_mm, pose.y_mm)
    head = to_pixels(pose.x_mm + length_mm * math.cos(pose.theta_rad),
                     pose.y_mm + length_mm * math.sin(pose.theta_rad))
    cv2.arrowedLine(panel, _point(tail), _point(head), _ORANGE, 2, cv2.LINE_AA, tipLength=0.3)
    cv2.circle(panel, _point(tail), 4, _ORANGE, -1, cv2.LINE_AA)


def _point(xy):
    return round(float(xy[0])), round(float(xy[1]))


def _fit_height(panel):
    scale = _PANEL_HEIGHT / panel.shape[0]
    return cv2.resize(panel, (round(panel.shape[1] * scale), _PANEL_HEIGHT),
                      interpolation=cv2.INTER_AREA)
