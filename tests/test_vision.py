import math

import numpy as np

from tymioplanet import grid, synthetic, visualize
from tymioplanet.datatypes import FREE, OCCUPIED
from tymioplanet.synthetic import GREEN, Shape
from tymioplanet.vision import VisionSystem


def _cell(cells, x_mm, y_mm):
    return cells[grid.world_to_cell(x_mm, y_mm)]


def test_demo_scene_end_to_end():
    scene = synthetic.demo_scene()
    observation = VisionSystem().update(synthetic.render(scene, noise_std=3.0))
    assert observation.camera_ok and observation.map_changed

    pose, truth = observation.pose, scene.robot
    assert math.hypot(pose.x_mm - truth.x_mm, pose.y_mm - truth.y_mm) < 10
    assert abs(pose.theta_rad - truth.theta_rad) < math.radians(3)

    world_map = observation.world_map
    assert world_map.grid.shape == (85, 119) and world_map.grid.dtype == np.int8
    assert len(world_map.obstacles) == 4  # the robot is not one of them
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


def test_covered_camera_keeps_the_last_map():
    vision_system = VisionSystem()
    frame = synthetic.render(synthetic.demo_scene())
    seen = vision_system.update(frame)
    covered = vision_system.update(np.zeros_like(frame))
    assert not covered.camera_ok and covered.pose is None and not covered.map_changed
    assert covered.world_map is seen.world_map
    assert vision_system.update(frame).camera_ok  # and it recovers


def test_camera_covered_from_the_start():
    observation = VisionSystem().update(np.zeros((720, 1280, 3), np.uint8))
    assert not observation.camera_ok and observation.world_map is None
    assert not VisionSystem().update(None).camera_ok


def test_hidden_tag_gives_no_pose_but_still_a_map():
    scene = synthetic.demo_scene()
    scene.robot = None
    observation = VisionSystem().update(synthetic.render(scene))
    assert observation.camera_ok and observation.pose is None
    assert len(observation.world_map.obstacles) == 4


def test_map_changed_only_when_the_scene_changes():
    scene = synthetic.demo_scene()
    vision_system = VisionSystem()
    assert vision_system.update(synthetic.render(scene, noise_std=3.0, seed=1)).map_changed
    assert not vision_system.update(synthetic.render(scene, noise_std=3.0, seed=2)).map_changed

    scene.shapes.append(Shape("circle", 700, 150, 50, GREEN))
    observation = vision_system.update(synthetic.render(scene, noise_std=3.0, seed=3))
    assert observation.map_changed and len(observation.world_map.waypoints) == 4


def test_live_view_renders_in_both_states():
    vision_system = VisionSystem()
    frame = synthetic.render(synthetic.demo_scene())
    observation = vision_system.update(frame)
    view = visualize.render_view(frame, vision_system, observation, [(340, 170), (1000, 200)])
    assert view.ndim == 3 and view.shape[1] > frame.shape[1] * 0.5
    black = np.zeros_like(frame)
    assert visualize.render_view(black, vision_system, vision_system.update(black)).ndim == 3
