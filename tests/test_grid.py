import numpy as np

from tymioplanet import grid
from tymioplanet.datatypes import FREE, OCCUPIED


def test_mask_to_grid_puts_row_zero_at_the_bottom():
    mask = np.zeros((20, 30), dtype=bool)
    mask[19, 0] = True  # bottom-left pixel of the top-down image = world origin
    cells = grid.mask_to_grid(mask, factor=5)
    assert cells.shape == (4, 6)
    assert cells[0, 0] == OCCUPIED
    assert (cells == OCCUPIED).sum() == 1


def test_one_pixel_is_enough_to_occupy_a_cell():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2, 7] = True
    assert grid.mask_to_grid(mask, factor=5)[1, 1] == OCCUPIED


def test_inflate_grows_by_the_radius():
    cells = np.full((41, 41), FREE, dtype=np.int8)
    cells[20, 20] = OCCUPIED
    inflated = grid.inflate(cells, radius_mm=50, cell_size_mm=10)
    assert inflated[20, 25] == OCCUPIED and inflated[20, 26] == FREE
    assert inflated[24, 23] == OCCUPIED  # 3-4-5 triangle: exactly 50 mm away
    assert inflated[24, 24] == FREE
    assert cells[20, 25] == FREE  # the input is left untouched


def test_gap_narrower_than_the_robot_closes_and_a_wide_one_stays_open():
    cells = np.full((60, 60), FREE, dtype=np.int8)
    cells[0:20, 10] = OCCUPIED
    cells[26:60, 10] = OCCUPIED  # 60 mm gap in a wall
    assert (grid.inflate(cells, 90, 10)[:, 10] == OCCUPIED).all()

    cells[26:46, 10] = FREE  # now a 260 mm gap
    assert grid.inflate(cells, 90, 10)[32, 10] == FREE


def test_inflate_without_obstacles_changes_nothing():
    cells = np.full((5, 5), FREE, dtype=np.int8)
    assert (grid.inflate(cells, 90, 10) == FREE).all()


def test_world_and_cell_conversions():
    assert grid.world_to_cell(25.0, 131.0, 10) == (13, 2)
    assert grid.cell_to_world(13, 2, 10) == (25.0, 135.0)
    cells = np.zeros((85, 119))
    assert grid.in_grid(cells, 84, 118) and not grid.in_grid(cells, 85, 0)
