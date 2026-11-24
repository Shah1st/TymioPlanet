"""Finds the objects on the sheet and sorts them into obstacles, planets and Earth.

Two steps, each in the colour space that suits it:
1. Object or sheet? Distance to the sheet's colour in Lab, where lightness (L) is separate
   from colour (a, b). Weighting L down makes shadows, which only change lightness, count
   less, while a black object still stands out.
2. Which object? Hue in HSV, which stays put when the lighting gets brighter or dimmer.
Thresholding RGB directly would mix brightness into every channel (course "Common
pitfalls": pick a colour space robust to illumination; Siegwart & Nourbakhsh 4.1.8.4).
"""

import math

import cv2
import numpy as np

from . import config
from .datatypes import HOME, OBSTACLE, WAYPOINT, Blob


def segment(top_down, valid, ignore=None):
    """{OBSTACLE: mask, WAYPOINT: mask, HOME: mask}, boolean masks over the top-down image.

    valid: pixels that are on the sheet. ignore: pixels to leave out (the robot).
    """
    usable = valid if ignore is None else valid & ~ignore
    # Blur first: thresholding amplifies pixel noise (course LAB 2, part 2).
    blurred = cv2.GaussianBlur(top_down, (config.BLUR_KERNEL, config.BLUR_KERNEL), 0)

    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB).astype(np.float32)
    if not usable.any():
        empty = np.zeros(valid.shape, dtype=bool)
        return {OBSTACLE: empty, WAYPOINT: empty.copy(), HOME: empty.copy()}
    sheet_colour = np.median(lab[usable], axis=0)  # the sheet is the most common colour
    difference = (lab - sheet_colour) * np.array([config.LIGHTNESS_WEIGHT, 1.0, 1.0], np.float32)
    is_object = (np.linalg.norm(difference, axis=2) > config.COLOR_THRESHOLD) & usable

    hue, saturation, value = cv2.split(cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV))
    coloured = (saturation >= config.MIN_SATURATION) & (value >= config.MIN_VALUE)
    green = coloured & (hue >= config.GREEN_HUE[0]) & (hue <= config.GREEN_HUE[1])
    blue = coloured & (hue >= config.BLUE_HUE[0]) & (hue <= config.BLUE_HUE[1])

    return {
        WAYPOINT: _clean(is_object & green),
        HOME: _clean(is_object & blue),
        OBSTACLE: _clean(is_object & ~green & ~blue),
    }


def find_blobs(mask, kind, view):
    """Blobs of one mask, in world millimetres, largest first."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for contour in contours:
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        # Centroid from image moments, as in the course LAB 2 part 1 solution.
        centre_px = [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]]
        x_mm, y_mm = view.top_down_to_world([centre_px])[0]
        area_mm2 = moments["m00"] * view.mm_per_pixel ** 2
        blobs.append(Blob(kind, float(x_mm), float(y_mm), math.sqrt(area_mm2 / math.pi)))
    return sorted(blobs, key=lambda blob: -blob.radius_mm)


def disc_mask(shape, centre_px, radius_px):
    """Boolean mask of a filled disc."""
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.circle(mask, (round(centre_px[0]), round(centre_px[1])), round(radius_px), 1, -1)
    return mask.astype(bool)


def _clean(mask):
    """Removes specks and fills holes.

    Opening (erode, dilate) deletes anything thinner than the kernel, such as the faint rim
    around a balloon; closing (dilate, erode) bridges small gaps. Then blobs below the
    minimum area are dropped and the rest are filled, which also covers highlights.
    """
    kernel = np.ones((config.MORPH_KERNEL, config.MORPH_KERNEL), np.uint8)
    cleaned = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area_px = config.MIN_BLOB_AREA_MM2 / config.MM_PER_PIXEL ** 2
    kept = [contour for contour in contours if cv2.contourArea(contour) >= min_area_px]
    filled = np.zeros(cleaned.shape, dtype=np.uint8)
    cv2.drawContours(filled, kept, -1, 1, cv2.FILLED)
    return filled.astype(bool)
