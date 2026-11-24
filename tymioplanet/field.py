"""Locates the field from its four corner tags and rectifies it into a top-down view.

A flat surface seen by a camera is related to its true shape by a homography (a 3x3
projective transform), fully determined by 4 point pairs: here the field's corners in the
image, marked by ArUco tags, and their known positions in millimetres. Once known, any
image pixel on the field converts to world millimetres, wherever the camera stands.

Each tag sits outside the field with one of its corners on a field corner. That inner
corner is the one used, so the tags themselves never appear on the map.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np

from . import config


@dataclass
class FieldView:
    """The calibrated relation between camera image, world (mm) and top-down image."""

    corners_px: np.ndarray  # 4x2 image corners of world (0,0), (W,0), (W,H), (0,H)
    image_to_world: np.ndarray  # 3x3 homography, image pixels -> world mm
    image_size: tuple  # (width, height) of the camera image in pixels
    width_mm: float = config.FIELD_WIDTH_MM
    height_mm: float = config.FIELD_HEIGHT_MM
    mm_per_pixel: float = config.MM_PER_PIXEL
    cell_size_mm: float = config.CELL_SIZE_MM

    @property
    def grid_shape(self):
        """(rows, cols) of the occupancy grid covering the field."""
        return (math.ceil(self.height_mm / self.cell_size_mm),
                math.ceil(self.width_mm / self.cell_size_mm))

    @property
    def top_down_size(self):
        """(width, height) in pixels of the top-down image: a whole number of pixels per cell."""
        factor = cells_factor(self.cell_size_mm, self.mm_per_pixel)
        rows, cols = self.grid_shape
        return cols * factor, rows * factor

    def to_world(self, points_px):
        """Image pixels (Nx2) -> world mm (Nx2)."""
        points = np.asarray(points_px, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(points, self.image_to_world).reshape(-1, 2)

    def world_to_top_down(self, points_mm):
        """World mm (Nx2) -> top-down pixels (Nx2). The top-down image has y pointing down."""
        points = np.asarray(points_mm, dtype=np.float64).reshape(-1, 2)
        height_px = self.top_down_size[1]
        return np.column_stack([points[:, 0] / self.mm_per_pixel,
                                height_px - points[:, 1] / self.mm_per_pixel])

    def top_down_to_world(self, points_px):
        """Top-down pixels (Nx2) -> world mm (Nx2)."""
        points = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
        height_px = self.top_down_size[1]
        return np.column_stack([points[:, 0] * self.mm_per_pixel,
                                (height_px - points[:, 1]) * self.mm_per_pixel])

    def warp(self, frame):
        """Top-down image of the field: x to the right, y up, mm_per_pixel per pixel."""
        width_px, height_px = self.top_down_size
        scale = 1.0 / self.mm_per_pixel
        world_to_top_down = np.array([[scale, 0.0, 0.0],
                                      [0.0, -scale, height_px],
                                      [0.0, 0.0, 1.0]])
        return cv2.warpPerspective(frame, world_to_top_down @ self.image_to_world,
                                   (width_px, height_px), flags=cv2.INTER_LINEAR)

    def valid_mask(self):
        """Top-down pixels that lie on the field, EDGE_MARGIN_MM away from its border."""
        width_px, height_px = self.top_down_size
        margin = config.EDGE_MARGIN_MM
        low = self.world_to_top_down([[margin, self.height_mm - margin]])[0]
        high = self.world_to_top_down([[self.width_mm - margin, margin]])[0]
        mask = np.zeros((height_px, width_px), dtype=bool)
        mask[math.ceil(low[1]):math.floor(high[1]), math.ceil(low[0]):math.floor(high[0])] = True
        return mask


def cells_factor(cell_size_mm, mm_per_pixel):
    """Top-down pixels per grid cell side."""
    factor = round(cell_size_mm / mm_per_pixel)
    if factor < 1 or abs(factor * mm_per_pixel - cell_size_mm) > 1e-9:
        raise ValueError("CELL_SIZE_MM must be a whole multiple of MM_PER_PIXEL")
    return factor


def find_field_corners(tags, corner_ids=config.CORNER_TAG_IDS):
    """Image corners (4x2) of the field in world order (0,0), (W,0), (W,H), (0,H), or None
    when a corner tag is missing.

    tags: {tag id: 4x2 corners} from markers.detect_tags. For each corner tag, the field
    corner is the tag corner nearest to the middle of the field, so it does not matter
    which way round a tag is stuck down.
    """
    if any(tag_id not in tags for tag_id in corner_ids):
        return None
    middle = np.mean([tags[tag_id].mean(axis=0) for tag_id in corner_ids], axis=0)
    corners = [min(tags[tag_id], key=lambda corner: np.linalg.norm(corner - middle))
               for tag_id in corner_ids]
    return np.array(corners, dtype=np.float32)


def calibrate(frame, tags):
    """FieldView from one frame in which the four corner tags are visible, or None."""
    corners_px = find_field_corners(tags)
    if corners_px is None:
        return None
    # Seen from above the world corners run counter-clockwise; in an image (y down) that
    # is a negative signed area. The opposite sign means two tags are swapped.
    if _signed_area(corners_px) >= 0:
        raise ValueError(
            f"corner tags are in mirrored order: seen from the camera, tags "
            f"{config.CORNER_TAG_IDS} must go bottom-left, bottom-right, top-right, top-left")
    world_mm = np.array([[0, 0], [config.FIELD_WIDTH_MM, 0],
                         [config.FIELD_WIDTH_MM, config.FIELD_HEIGHT_MM],
                         [0, config.FIELD_HEIGHT_MM]], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(corners_px, world_mm)
    return FieldView(corners_px, homography, (frame.shape[1], frame.shape[0]))


def _signed_area(quad):
    x, y = quad[:, 0], quad[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
