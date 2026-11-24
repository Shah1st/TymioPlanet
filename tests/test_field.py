import numpy as np
import pytest

from tymioplanet import config, field, markers, synthetic


def _calibrate(frame):
    return field.calibrate(frame, markers.detect_tags(frame))


def test_field_corners_found_from_the_corner_tags():
    truth = synthetic.default_field_corners()
    view = _calibrate(synthetic.render(synthetic.demo_scene(), noise_std=3.0))
    assert view is not None
    assert np.abs(view.corners_px - truth).max() < 2.0  # pixels


def test_image_corners_map_to_field_corners_in_mm():
    truth = synthetic.default_field_corners()
    view = _calibrate(synthetic.render(synthetic.Scene()))
    w, h = config.FIELD_WIDTH_MM, config.FIELD_HEIGHT_MM
    assert np.abs(view.to_world(truth) - [[0, 0], [w, 0], [w, h], [0, h]]).max() < 3.0


def test_top_down_view_is_upright():
    # A blue disc near the world origin must appear bottom-left in the top-down image.
    scene = synthetic.Scene(shapes=[synthetic.Shape("circle", 150, 100, 50, synthetic.BLUE)])
    frame = synthetic.render(scene)
    view = _calibrate(frame)
    top_down = view.warp(frame)
    u, v = view.world_to_top_down([[150, 100]])[0]
    assert tuple(top_down[round(v), round(u)]) == synthetic.BLUE
    assert u < top_down.shape[1] / 2 and v > top_down.shape[0] / 2
    assert np.allclose(view.top_down_to_world([[u, v]]), [[150, 100]])


def test_the_corner_tags_stay_off_the_map():
    frame = synthetic.render(synthetic.Scene())
    view = _calibrate(frame)
    inside = view.warp(frame)[view.valid_mask()]
    assert inside.min() > 200  # only white floor: no black tag pixel


def test_no_field_when_a_corner_tag_is_missing():
    frame = synthetic.render(synthetic.Scene(), corner_tags=(1, 2, 3, None))
    assert _calibrate(frame) is None
    assert _calibrate(np.zeros((720, 1280, 3), np.uint8)) is None


def test_swapped_corner_tags_are_reported():
    frame = synthetic.render(synthetic.Scene(), corner_tags=(2, 1, 3, 4))
    with pytest.raises(ValueError, match="mirrored"):
        _calibrate(frame)
