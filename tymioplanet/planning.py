"""PLACEHOLDER (global navigation owner): route through all planets and back to Earth."""

from .datatypes import Pose, WorldMap


def plan_route(world_map: WorldMap, pose: Pose) -> list[tuple[float, float]]:
    """Path from the robot to every waypoint, then home, as world points (x_mm, y_mm).

    Inputs:
      world_map.grid_inflated  int8 [row, col], 0 free, -1 obstacle, already grown by the
                               robot radius: plan for a point robot. Same convention as the
                               course LAB 5 `grid_search`, which runs on it unchanged.
      world_map.waypoints      green balloons (Blob), in free cells
      world_map.home           blue balloon (Blob) or None
      pose                     current robot pose
    Convert with tymioplanet.grid.world_to_cell / cell_to_world.

    Called again whenever Observation.map_changed is True, so the visiting order may change.
    """
    raise NotImplementedError
