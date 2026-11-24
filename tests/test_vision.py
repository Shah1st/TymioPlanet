import math

import numpy as np

from tymioplanet import grid, synthetic, visualize
from tymioplanet.datatypes import FREE, OCCUPIED, Pose
from tymioplanet.synthetic import GREEN, RED, Shape
from tymioplanet.vision import VisionSystem


def _cell(cells, x_mm, y_mm):
    return cells[grid.world_to_cell(x_mm, y_mm)]


def test_demo_scene_end_to_end():
    scene = synthetic.demo_scene()
    observation = VisionSystem().update(synthetic.render(scene, noise_std=3.0))
    assert observation.camera_ok

    pose, truth = observation.pose, scene.robot
    assert math.hypot(pose.x_mm - truth.x_mm, pose.y_mm - truth.y_mm) < 10
    assert abs(pose.theta_rad - truth.theta_rad) < math.radians(3)

    world_map = observation.world_map
    assert world_map.grid.shape == (85, 119) and world_map.grid.dtype == np.int8
    assert len(world_map.obstacles) == 4  # neither the robot nor the corner tags
    assert len(world_map.waypoints) == 3
    assert math.hypot(world_map.home.x_mm - 150, world_map.home.y_mm - 150) < 10

    assert _cell(world_map.grid, 550, 250) == OCCUPIED  # red balloon
    assert _cell(world_map.grid, 330, 400) == OCCUPIED  # black square
    for target in world_map.waypoints + [world_map.home]:
        assert _cell(world_map.grid, target.x_mm, target.y_mm) == FREE


def test_the_trap_between_two_asteroids_is_closed_only_in_the_inflated_grid():
    world_map = VisionSystem().update(synthetic.render(synthetic.demo_scene())).world_map
    assert _cell(world_map.grid, 550, 340) == FREE
    assert _cell(world_map.grid_inflated, 550, 340) == OCCUPIED
    assert _cell(world_map.grid_inflated, 800, 150) == FREE  # open space stays open


def test_the_map_is_made_once_and_then_only_the_robot_is_tracked():
    scene = synthetic.demo_scene()
    vision_system = VisionSystem()
    first = vision_system.update(synthetic.render(scene))

    scene.shapes.append(Shape("circle", 700, 150, 50, RED))  # obstacle added afterwards
    scene.robot = Pose(900, 400, math.radians(-90))  # and the robot has moved
    later = vision_system.update(synthetic.render(scene))

    assert later.world_map is first.world_map and len(later.world_map.obstacles) == 4
    assert math.hypot(later.pose.x_mm - 900, later.pose.y_mm - 400) < 10
    assert abs(later.pose.theta_rad - math.radians(-90)) < math.radians(3)


def test_snap_retakes_the_map():
    scene = synthetic.demo_scene()
    vision_system = VisionSystem()
    vision_system.update(synthetic.render(scene))
    scene.shapes.append(Shape("circle", 700, 150, 50, GREEN))
    world_map = vision_system.snap(synthetic.render(scene))
    assert len(world_map.waypoints) == 4 and vision_system.world_map is world_map


def test_covered_camera_gives_no_pose_and_keeps_the_map():
    vision_system = VisionSystem()
    frame = synthetic.render(synthetic.demo_scene())
    seen = vision_system.update(frame)
    for covered_frame in (np.zeros_like(frame), None):
        covered = vision_system.update(covered_frame)
        assert not covered.camera_ok and covered.pose is None
        assert covered.world_map is seen.world_map
    assert vision_system.update(frame).pose is not None  # and it recovers


def test_camera_covered_from_the_start_then_uncovered():
    vision_system = VisionSystem()
    blind = vision_system.update(np.zeros((720, 1280, 3), np.uint8))
    assert not blind.camera_ok and blind.world_map is None
    assert vision_system.update(synthetic.render(synthetic.demo_scene())).world_map is not None


def test_no_map_until_all_four_corner_tags_are_visible():
    scene = synthetic.demo_scene()
    vision_system = VisionSystem()
    partial = vision_system.update(synthetic.render(scene, corner_tags=(1, 2, None, 4)))
    assert partial.camera_ok and partial.world_map is None and partial.pose is None
    assert vision_system.update(synthetic.render(scene)).world_map is not None


def test_hidden_robot_tag_gives_no_pose_but_still_a_map():
    scene = synthetic.demo_scene()
    scene.robot = None
    observation = VisionSystem().update(synthetic.render(scene))
    assert observation.camera_ok and observation.pose is None
    assert len(observation.world_map.obstacles) == 4


def test_live_view_renders_in_every_state():
    vision_system = VisionSystem()
    black = np.zeros((720, 1280, 3), np.uint8)
    assert visualize.render_view(black, vision_system, vision_system.update(black)).ndim == 3
    frame = synthetic.render(synthetic.demo_scene())
    observation = vision_system.update(frame)
    view = visualize.render_view(frame, vision_system, observation, [(340, 170), (1000, 200)])
    assert view.ndim == 3 and view.shape[1] > frame.shape[1]
    assert visualize.render_view(black, vision_system, vision_system.update(black)).ndim == 3
