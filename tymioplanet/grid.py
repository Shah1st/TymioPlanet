"""Occupancy grid: the field cut into square cells, each free or occupied.

A fixed grid suits this scene because it takes any obstacle shape straight from the
segmentation mask, and a new obstacle is just more occupied cells. Its known weakness,
narrow passages lost to coarse cells, is kept small by the 1 cm cell (course Navigation
slides, fixed cell decomposition).

Values follow the course A*/Dijkstra solution (LAB 5): 0 free, -1 obstacle.
"""

import cv2
import numpy as np

from . import config
from .datatypes import FREE, OCCUPIED


def mask_to_grid(obstacle_mask, factor):
    """Grid from a top-down obstacle mask; factor = top-down pixels per cell side.

    A cell is occupied as soon as one of its pixels is. The mask has y pointing down, the
    grid has row 0 at y = 0, hence the flip.
    """
    rows, cols = obstacle_mask.shape[0] // factor, obstacle_mask.shape[1] // factor
    cells = np.flipud(obstacle_mask).reshape(rows, factor, cols, factor).any(axis=(1, 3))
    grid = np.full((rows, cols), FREE, dtype=np.int8)
    grid[cells] = OCCUPIED
    return grid


def inflate(grid, radius_mm, cell_size_mm=config.CELL_SIZE_MM):
    """Grows every obstacle by radius_mm.

    Path planners treat the robot as a point (course Navigation slides: "a priori expansion
    of obstacles"). Growing the obstacles by the robot's radius makes that true: wherever
    the point may go, the real robot fits. A gap narrower than the robot closes.
    """
    occupied = grid == OCCUPIED
    inflated = grid.copy()
    if radius_mm <= 0 or not occupied.any():
        return inflated
    # Distance, in cells, from every cell to the nearest occupied one.
    distance = cv2.distanceTransform((~occupied).astype(np.uint8), cv2.DIST_L2,
                                     cv2.DIST_MASK_PRECISE)
    inflated[distance * cell_size_mm <= radius_mm] = OCCUPIED
    return inflated


def world_to_cell(x_mm, y_mm, cell_size_mm=config.CELL_SIZE_MM):
    """(row, col) of the cell containing a world point."""
    return int(y_mm // cell_size_mm), int(x_mm // cell_size_mm)


def cell_to_world(row, col, cell_size_mm=config.CELL_SIZE_MM):
    """World (x_mm, y_mm) of a cell's centre."""
    return (col + 0.5) * cell_size_mm, (row + 0.5) * cell_size_mm


def in_grid(grid, row, col):
    return 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]
