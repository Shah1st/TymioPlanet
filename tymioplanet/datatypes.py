"""Data passed between the modules. This is the interface the whole team codes against.

World frame: on the field, millimetres, origin at the inner corner of corner tag 1, x towards
tag 2, y towards tag 4, so that the frame is right-handed seen from above.
Heading: radians, counter-clockwise from +x, in (-pi, pi].
"""

from dataclasses import dataclass, field

import numpy as np

OBSTACLE = "obstacle"
WAYPOINT = "waypoint"  # green balloon, a planet to reach
HOME = "home"  # blue balloon, Earth: start and final goal

# Grid cell values, as in the course A*/Dijkstra solution (LAB 5): values above 0 are free
# cells with an extra cost.
FREE = 0
OCCUPIED = -1


@dataclass(frozen=True)
class Pose:
    x_mm: float
    y_mm: float
    theta_rad: float


@dataclass(frozen=True)
class Blob:
    """One object seen on the field, approximated by a disc of equal area."""

    kind: str  # OBSTACLE, WAYPOINT or HOME
    x_mm: float
    y_mm: float
    radius_mm: float


@dataclass
class WorldMap:
    """The map handed to the path planner.

    grid and grid_inflated are int8 arrays indexed [row, col]; row 0 is y = 0, col 0 is x = 0.
    Use tymioplanet.grid.world_to_cell / cell_to_world instead of indexing by hand.
    grid_inflated has every obstacle grown by the robot radius, so the planner can treat the
    robot as a point. Waypoints and home are never marked as occupied.
    """

    grid: np.ndarray
    grid_inflated: np.ndarray
    cell_size_mm: float
    obstacles: list[Blob] = field(default_factory=list)
    waypoints: list[Blob] = field(default_factory=list)
    home: Blob | None = None


@dataclass
class Observation:
    """What the camera says for one frame.

    camera_ok is False when no tag at all is visible (camera covered). pose is None then,
    and also when only the robot tag is hidden. world_map is the map built from the one
    snapshot; it does not change afterwards (None until the snapshot is taken).
    """

    camera_ok: bool
    pose: Pose | None
    world_map: WorldMap | None
