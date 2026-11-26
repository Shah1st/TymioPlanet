import math

from tymioplanet import colors, field, markers, synthetic
from tymioplanet.datatypes import HOME, OBSTACLE, WAYPOINT
from tymioplanet.synthetic import BLACK, BLUE, GREEN, RED, Scene, Shape


def _blobs(scene):
    frame = synthetic.render(scene, noise_std=3.0)
    view = field.calibrate(frame, markers.detect_tags(frame))
    masks = colors.segment(view.warp(frame), view.valid_mask())
    return {kind: colors.find_blobs(masks[kind], kind, view) for kind in masks}


def test_each_colour_goes_to_its_class():
    blobs = _blobs(Scene(shapes=[
        Shape("circle", 300, 300, 60, RED),
        Shape("square", 600, 300, 100, BLACK),
        Shape("circle", 750, 300, 50, GREEN),
        Shape("circle", 600, 500, 55, BLUE),
    ]))
    assert len(blobs[OBSTACLE]) == 2 and len(blobs[WAYPOINT]) == 1 and len(blobs[HOME]) == 1

    green, blue = blobs[WAYPOINT][0], blobs[HOME][0]
    assert math.hypot(green.x_mm - 750, green.y_mm - 300) < 10
    assert abs(green.radius_mm - 50) < 5
    assert math.hypot(blue.x_mm - 600, blue.y_mm - 500) < 10

    centres = sorted((round(b.x_mm, -1), round(b.y_mm, -1)) for b in blobs[OBSTACLE])
    assert centres == [(300, 300), (600, 300)]


def test_touching_balloons_of_different_colours_stay_separate():
    blobs = _blobs(Scene(shapes=[
        Shape("circle", 500, 400, 50, RED),
        Shape("circle", 595, 400, 50, GREEN),
    ]))
    assert len(blobs[OBSTACLE]) == 1 and len(blobs[WAYPOINT]) == 1
    assert abs(blobs[WAYPOINT][0].x_mm - 595) < 10


def test_empty_field_has_no_objects():
    blobs = _blobs(Scene())
    assert not any(blobs.values())


def test_a_soft_shadow_is_not_an_obstacle():
    blobs = _blobs(Scene(shapes=[Shape("circle", 600, 400, 120, (200, 200, 200))]))
    assert not blobs[OBSTACLE]
