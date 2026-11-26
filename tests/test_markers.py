import math

import numpy as np
import pytest

from tymioplanet import config, field, markers, synthetic
from tymioplanet.datatypes import Pose


def _angle_error(a, b):
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


@pytest.mark.parametrize("x, y, degrees", [
    (300, 200, 0), (600, 420, 90), (700, 450, 180), (200, 480, -135), (740, 150, 33),
])
def test_pose_from_tag(x, y, degrees):
    truth = Pose(x, y, math.radians(degrees))
    frame = synthetic.render(synthetic.Scene(robot=truth), noise_std=3.0)
    view = field.calibrate(frame, markers.detect_tags(frame))
    corners = markers.detect_tags(frame)[config.ROBOT_TAG_ID]
    pose = markers.pose_from_tag(corners, view)
    assert math.hypot(pose.x_mm - x, pose.y_mm - y) < 10.0
    assert _angle_error(pose.theta_rad, truth.theta_rad) < math.radians(3)
    assert -math.pi < pose.theta_rad <= math.pi


def test_no_robot_tag_no_robot_detection():
    tags = markers.detect_tags(synthetic.render(synthetic.Scene()))
    assert sorted(tags) == list(config.CORNER_TAG_IDS)  # only the four corner tags


def test_tag_offset_moves_the_pose_back_along_the_heading():
    truth = Pose(600, 400, math.radians(90))
    frame = synthetic.render(synthetic.Scene(robot=truth))
    view = field.calibrate(frame, markers.detect_tags(frame))
    corners = markers.detect_tags(frame)[config.ROBOT_TAG_ID]
    pose = markers.pose_from_tag(corners, view, tag_offset_mm=(20.0, 0.0))
    assert pose.x_mm == pytest.approx(600, abs=5)
    assert pose.y_mm == pytest.approx(380, abs=5)


def test_parallax_correction_pulls_points_towards_the_nadir():
    # Camera 1 m up, tag 100 mm up: a point seen 500 mm from the nadir stands at 450 mm.
    corrected = markers.correct_parallax([[500.0, 0.0]], [0.0, 0.0], 1000.0, 100.0)
    assert np.allclose(corrected, [[450.0, 0.0]])
