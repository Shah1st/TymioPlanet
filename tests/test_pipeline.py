"""The parts together: camera frame -> map and pose -> route -> path follower."""

import math

from tymioplanet import config, control, grid, planning, synthetic
from tymioplanet.datatypes import FREE
from tymioplanet.vision import VisionSystem


def _observe():
    return VisionSystem().update(synthetic.render(synthetic.demo_scene()))


def test_route_visits_every_planet_and_ends_on_earth():
    observation = _observe()
    world_map = observation.world_map
    route = planning.plan_route(world_map, observation.pose)

    for planet in world_map.waypoints:
        assert min(math.hypot(x - planet.x_mm, y - planet.y_mm) for x, y in route) < 20
    # Earth sits closer to the border than the robot may go: the route ends next to it.
    end_x, end_y = route[-1]
    assert math.hypot(end_x - world_map.home.x_mm, end_y - world_map.home.y_mm) < 60
    for x, y in route:
        assert world_map.grid[grid.world_to_cell(x, y)] == FREE


def test_follower_drives_the_route_without_touching_an_obstacle():
    observation = _observe()
    world_map = observation.world_map
    route = planning.plan_route(world_map, observation.pose)

    x, y, theta = observation.pose.x_mm, observation.pose.y_mm, observation.pose.theta_rad
    index, done, dt = 0, False, 0.1
    for _ in range(3000):
        left, right, index, done = control.follow_path_step((x, y, theta), route, index)
        if done:
            break
        v_left, v_right = (left / config.THYMIO_UNITS_PER_MM_S, right / config.THYMIO_UNITS_PER_MM_S)
        theta += (v_right - v_left) / config.WHEELBASE_MM * dt
        x += (v_left + v_right) / 2 * math.cos(theta) * dt
        y += (v_left + v_right) / 2 * math.sin(theta) * dt
        for blob in world_map.obstacles:  # the body (80 mm) never reaches an obstacle
            assert math.hypot(x - blob.x_mm, y - blob.y_mm) > blob.radius_mm + config.ROBOT_RADIUS_MM
    assert done
