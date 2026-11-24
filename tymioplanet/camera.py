"""Where frames come from: a live camera, a saved picture, or the synthetic scene."""

import cv2
import numpy as np

from . import synthetic


class LiveCamera:
    """A webcam (index) or a video file / stream (path or URL)."""

    def __init__(self, source):
        self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            raise RuntimeError(f"cannot open camera source {source!r}")
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always the newest frame, not a backlog

    def read(self):
        ok, frame = self.capture.read()
        return frame if ok else None

    def release(self):
        self.capture.release()


class ImageCamera:
    """The same saved picture on every read."""

    def __init__(self, path):
        self.frame = cv2.imread(path)
        if self.frame is None:
            raise RuntimeError(f"cannot read image {path!r}")

    def read(self):
        return self.frame.copy()

    def release(self):
        pass


class SyntheticCamera:
    """Renders a Scene; the scene can be edited and the lens covered while running."""

    def __init__(self, scene=None):
        self.scene = scene or synthetic.demo_scene()
        self.covered = False
        self._frame = None
        self._rendered = None  # what the cached frame was rendered from

    def read(self):
        state = (tuple(self.scene.shapes), self.scene.robot)
        if state != self._rendered:
            self._frame = synthetic.render(self.scene, noise_std=3.0)
            self._rendered = state
        if self.covered:
            return np.zeros_like(self._frame)
        return self._frame.copy()

    def release(self):
        pass


def open_source(source):
    """'synthetic', a camera index such as '0', or a path to a picture or video."""
    if source == "synthetic":
        return SyntheticCamera()
    if source.isdigit():
        return LiveCamera(int(source))
    if source.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
        return ImageCamera(source)
    return LiveCamera(source)
