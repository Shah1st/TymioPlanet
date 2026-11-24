import numpy as np

from tymioplanet import config, sheet, synthetic


def test_corners_found_under_perspective():
    truth = synthetic.default_sheet_corners()
    view = sheet.calibrate(synthetic.render(synthetic.demo_scene(), noise_std=3.0))
    assert view is not None
    assert np.abs(view.corners_px - truth).max() < 3.0  # pixels


def test_image_corners_map_to_sheet_corners_in_mm():
    truth = synthetic.default_sheet_corners()
    view = sheet.calibrate(synthetic.render(synthetic.Scene()))
    w, h = config.SHEET_WIDTH_MM, config.SHEET_HEIGHT_MM
    assert np.abs(view.to_world(truth) - [[0, 0], [w, 0], [w, h], [0, h]]).max() < 5.0


def test_x_axis_follows_the_long_side_when_sheet_is_upright_in_the_image():
    corners = np.float32([[650, 1200], [640, 90], [70, 80], [60, 1190]])
    frame = synthetic.render(synthetic.Scene(), image_size=(720, 1280), sheet_corners=corners)
    view = sheet.calibrate(frame)
    along_x = np.linalg.norm(view.corners_px[1] - view.corners_px[0])
    along_y = np.linalg.norm(view.corners_px[3] - view.corners_px[0])
    assert along_x > along_y


def test_top_down_view_is_upright():
    # A blue disc near the world origin must appear bottom-left in the top-down image.
    scene = synthetic.Scene(shapes=[synthetic.Shape("circle", 150, 100, 50, synthetic.BLUE)])
    view = sheet.calibrate(synthetic.render(scene))
    top_down = view.warp(synthetic.render(scene))
    u, v = view.world_to_top_down([[150, 100]])[0]
    assert tuple(top_down[round(v), round(u)]) == synthetic.BLUE
    assert u < top_down.shape[1] / 2 and v > top_down.shape[0] / 2
    assert np.allclose(view.top_down_to_world([[u, v]]), [[150, 100]])


def test_no_sheet_in_a_dark_or_blank_frame():
    assert sheet.find_sheet_corners(np.zeros((720, 1280, 3), np.uint8)) is None
    assert sheet.find_sheet_corners(np.full((720, 1280, 3), 60, np.uint8)) is None
