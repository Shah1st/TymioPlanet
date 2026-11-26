"""Where frames come from: a live camera, a saved picture, or the synthetic scene."""

import threading
import time

import cv2
import numpy as np

from . import synthetic


class LiveCamera:
    """A webcam (index) or a video file / stream (path or URL).

    A webcam is read by a background thread that keeps only the newest frame. The driver
    queues frames, so reading them one by one in a loop slower than the camera shows the
    past: the picture, and the robot pose with it, falls further and further behind.
    """

    def __init__(self, source):
        self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            raise RuntimeError(f"cannot open camera source {source!r}")
        self._live = isinstance(source, int)  # a file must be read frame by frame instead
        if self._live:
            self._frame = None
            self._fresh = threading.Condition()
            self._running = True
            self._thread = threading.Thread(target=self._grab, daemon=True)
            self._thread.start()

    def _grab(self):
        while self._running:
            ok, frame = self.capture.read()
            with self._fresh:
                self._frame = frame if ok else None
                self._fresh.notify_all()
            if not ok:
                time.sleep(0.01)

    def read(self):
        """The newest frame; waits for the next one if this one was already given out."""
        if not self._live:
            ok, frame = self.capture.read()
            return frame if ok else None
        with self._fresh:
            self._fresh.wait(timeout=1.0)
            frame, self._frame = self._frame, None
        return frame

    def release(self):
        if self._live:
            self._running = False
            self._thread.join(timeout=2.0)
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
