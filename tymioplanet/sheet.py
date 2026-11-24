"""Finds the white sheet in the camera image and rectifies it into a top-down view.

A flat sheet seen by a camera is related to its true shape by a homography (a 3x3 projective
transform), fully determined by 4 point pairs: here the sheet's corners in the image and
their known positions in millimetres. Once known, any image pixel on the sheet converts to
world millimetres, wherever the camera stands.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np

from . import config


@dataclass
class SheetView:
    """The calibrated relation between camera image, world (mm) and top-down image."""

    corners_px: np.ndarray  # 4x2 image corners of world (0,0), (W,0), (W,H), (0,H)
    image_to_world: np.ndarray  # 3x3 homography, image pixels -> world mm
    brightness: float  # median grey level of the sheet at calibration
    image_size: tuple  # (width, height) of the camera image in pixels
    width_mm: float = config.SHEET_WIDTH_MM
    height_mm: float = config.SHEET_HEIGHT_MM
    mm_per_pixel: float = config.MM_PER_PIXEL
    cell_size_mm: float = config.CELL_SIZE_MM

    @property
    def grid_shape(self):
        """(rows, cols) of the occupancy grid covering the sheet."""
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
        """Top-down image of the sheet: x to the right, y up, mm_per_pixel per pixel."""
        width_px, height_px = self.top_down_size
        scale = 1.0 / self.mm_per_pixel
        world_to_top_down = np.array([[scale, 0.0, 0.0],
                                      [0.0, -scale, height_px],
                                      [0.0, 0.0, 1.0]])
        return cv2.warpPerspective(frame, world_to_top_down @ self.image_to_world,
                                   (width_px, height_px), flags=cv2.INTER_LINEAR)

    def valid_mask(self):
        """Top-down pixels that lie on the sheet, EDGE_MARGIN_MM away from its border."""
        width_px, height_px = self.top_down_size
        margin = config.EDGE_MARGIN_MM
        low = self.world_to_top_down([[margin, self.height_mm - margin]])[0]
        high = self.world_to_top_down([[self.width_mm - margin, margin]])[0]
        mask = np.zeros((height_px, width_px), dtype=bool)
        mask[math.ceil(low[1]):math.floor(high[1]), math.ceil(low[0]):math.floor(high[0])] = True
        return mask

    def sheet_brightness(self, frame):
        """Median grey level inside the calibrated sheet outline in this frame."""
        return _median_inside(_grey(frame), self.corners_px)


def cells_factor(cell_size_mm, mm_per_pixel):
    """Top-down pixels per grid cell side."""
    factor = round(cell_size_mm / mm_per_pixel)
    if factor < 1 or abs(factor * mm_per_pixel - cell_size_mm) > 1e-9:
        raise ValueError("CELL_SIZE_MM must be a whole multiple of MM_PER_PIXEL")
    return factor


def find_sheet_corners(frame):
    """Image corners (4x2) of the white sheet, ordered top-left, top-right, bottom-right,
    bottom-left as seen in the image, or None when no sheet is visible.

    Blur, then Otsu's threshold on brightness (the sheet is the bright class, the table the
    dark one), then the largest contour. Its convex hull is taken so that the robot or a
    balloon standing on the border does not dent the outline, and the hull is simplified
    until 4 corners remain.
    """
    grey = cv2.GaussianBlur(_grey(frame), (config.BLUR_KERNEL, config.BLUR_KERNEL), 0)
    _, bright = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    if cv2.contourArea(hull) < config.MIN_SHEET_AREA_FRACTION * grey.size:
        return None
    quad = _four_corners(hull)
    if quad is None:
        return None
    if cv2.contourArea(quad) < 0.9 * cv2.contourArea(hull):
        return None  # the bright region is not a quadrilateral
    if _median_inside(grey, quad) < config.MIN_SHEET_BRIGHTNESS:
        return None
    return _order_corners(quad)


def calibrate(frame):
    """SheetView from one frame in which the whole sheet is visible, or None."""
    corners = find_sheet_corners(frame)
    if corners is None:
        return None
    top_left, top_right, bottom_right, bottom_left = corners
    # World corners (0,0), (W,0), (W,H), (0,H), counter-clockwise seen from above.
    # x runs along the sheet's long side, whichever way the sheet lies in the image.
    if np.linalg.norm(top_right - top_left) >= np.linalg.norm(bottom_left - top_left):
        world_order = [bottom_left, bottom_right, top_right, top_left]
    else:
        world_order = [bottom_right, top_right, top_left, bottom_left]
    corners_px = np.array(world_order, dtype=np.float32)
    world_mm = np.array([[0, 0], [config.SHEET_WIDTH_MM, 0],
                         [config.SHEET_WIDTH_MM, config.SHEET_HEIGHT_MM],
                         [0, config.SHEET_HEIGHT_MM]], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(corners_px, world_mm)
    return SheetView(corners_px, homography, _median_inside(_grey(frame), corners_px),
                     (frame.shape[1], frame.shape[0]))


def _grey(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame


def _four_corners(hull):
    """Simplifies a convex outline until it has exactly 4 corners (4x2), or None."""
    perimeter = cv2.arcLength(hull, True)
    for tolerance in np.arange(0.01, 0.11, 0.01):
        approx = cv2.approxPolyDP(hull, tolerance * perimeter, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
        if len(approx) < 4:
            return None
    return None


def _order_corners(quad):
    """Top-left, top-right, bottom-right, bottom-left (image axes: x right, y down)."""
    sums = quad.sum(axis=1)  # x + y: smallest at top-left, largest at bottom-right
    diffs = quad[:, 0] - quad[:, 1]  # x - y: largest at top-right, smallest at bottom-left
    return np.array([quad[np.argmin(sums)], quad[np.argmax(diffs)],
                     quad[np.argmax(sums)], quad[np.argmin(diffs)]], dtype=np.float32)


def _median_inside(grey, quad):
    mask = np.zeros(grey.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(quad).astype(np.int32), 255)
    return float(np.median(grey[mask > 0]))
