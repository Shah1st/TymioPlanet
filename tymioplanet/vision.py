"""The vision module: one camera frame in, robot pose and map out."""

import math

from . import colors, config, grid, markers, sheet
from .datatypes import HOME, OBSTACLE, WAYPOINT, Observation, WorldMap


class VisionSystem:
    """Keeps the sheet calibration and the last map between frames.

    The camera is fixed, so the sheet is located once (first frame where it is fully
    visible) and reused; call recalibrate() if the camera or the sheet is moved.
    """

    def __init__(self):
        self.view = None  # SheetView, set by the first successful calibration
        self.world_map = None  # last map built
        self.top_down = None  # last rectified image, for display
        self._last_pose = None

    def recalibrate(self):
        self.view = None

    def update(self, frame):
        """Observation for one BGR frame (None counts as a covered camera)."""
        if frame is None:
            return self._blind()
        if self.view is None:
            self.view = sheet.calibrate(frame)
            if self.view is None:
                return self._blind()
        # A covered camera shows a dark image where the sheet used to be.
        if self.view.sheet_brightness(frame) < config.COVERED_BRIGHTNESS_RATIO * self.view.brightness:
            return self._blind()

        pose = None
        corners = markers.detect_tags(frame).get(config.ROBOT_TAG_ID)
        if corners is not None:
            pose = markers.pose_from_tag(corners, self.view)
            self._last_pose = pose

        self.top_down = self.view.warp(frame)
        new_map = self._build_map(self.top_down)
        changed = self.world_map is None or _maps_differ(self.world_map, new_map)
        self.world_map = new_map
        return Observation(True, pose, new_map, changed)

    def _blind(self):
        return Observation(False, None, self.world_map, False)

    def _build_map(self, top_down):
        view = self.view
        valid = view.valid_mask()
        # The robot is not an obstacle. If its tag is hidden this frame, it is still close
        # to where it was last seen.
        ignore = None
        if self._last_pose is not None:
            centre = view.world_to_top_down([[self._last_pose.x_mm, self._last_pose.y_mm]])[0]
            ignore = colors.disc_mask(valid.shape, centre,
                                      config.ROBOT_MASK_RADIUS_MM / view.mm_per_pixel)
        masks = colors.segment(top_down, valid, ignore)

        factor = sheet.cells_factor(view.cell_size_mm, view.mm_per_pixel)
        raw = grid.mask_to_grid(masks[OBSTACLE], factor)
        inflated = grid.inflate(raw, config.ROBOT_RADIUS_MM + config.SAFETY_MARGIN_MM,
                                view.cell_size_mm)
        homes = colors.find_blobs(masks[HOME], HOME, view)
        return WorldMap(
            grid=raw,
            grid_inflated=inflated,
            cell_size_mm=view.cell_size_mm,
            obstacles=colors.find_blobs(masks[OBSTACLE], OBSTACLE, view),
            waypoints=colors.find_blobs(masks[WAYPOINT], WAYPOINT, view),
            home=homes[0] if homes else None,  # the largest blue blob
        )


def _maps_differ(old, new, tolerance_mm=config.MAP_CHANGE_TOLERANCE_MM):
    """True when an obstacle or target appeared, disappeared or moved beyond the tolerance."""
    old_home = [old.home] if old.home else []
    new_home = [new.home] if new.home else []
    return (_blobs_differ(old.obstacles, new.obstacles, tolerance_mm)
            or _blobs_differ(old.waypoints, new.waypoints, tolerance_mm)
            or _blobs_differ(old_home, new_home, tolerance_mm))


def _blobs_differ(old, new, tolerance_mm):
    if len(old) != len(new):
        return True
    remaining = list(new)
    for blob in old:
        nearest = min(remaining, key=lambda other: _distance(blob, other))
        if _distance(blob, nearest) > tolerance_mm:
            return True
        remaining.remove(nearest)
    return False


def _distance(a, b):
    return math.hypot(a.x_mm - b.x_mm, a.y_mm - b.y_mm)
