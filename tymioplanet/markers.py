"""Robot pose from the ArUco tag on top of the Thymio.

An ArUco tag is a black square holding a small grid of black and white cells that encodes
an id. The detector finds dark quadrilaterals, rectifies each one, reads its cells and
keeps those matching a dictionary entry. It returns the 4 corners in a fixed order
(top-left, top-right, bottom-right, bottom-left of the upright tag), so one tag gives both
position (the centre) and heading (the direction of its top edge). Template matching
cannot do this: it is neither rotation nor scale invariant (course LAB 2, part 2).
"""

import math
import os
import zlib

import cv2
import numpy as np

from . import config
from .datatypes import Pose


def detect_tags(frame, dictionary_name=config.ARUCO_DICTIONARY):
    """{tag id: 4x2 corners in image pixels} for every tag found in the frame."""
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    corners, ids, _ = cv2.aruco.ArucoDetector(dictionary, parameters).detectMarkers(grey)
    if ids is None:
        return {}
    return {int(tag_id): np.asarray(quad, dtype=np.float64).reshape(4, 2)
            for tag_id, quad in zip(ids.flatten(), corners)}


def pose_from_tag(corners_px, view, camera_height_mm=config.CAMERA_HEIGHT_MM,
                  tag_height_mm=config.TAG_HEIGHT_MM, tag_offset_mm=config.TAG_OFFSET_MM):
    """Pose of the robot (middle of its wheel axis) in the world frame.

    corners_px: the tag's 4 corners in the image. view: the calibrated FieldView.
    The tag centre and the middle of its top edge are converted to millimetres; the heading
    is the direction from the first to the second.
    """
    corners = np.asarray(corners_px, dtype=np.float64).reshape(4, 2)
    centre_px = corners.mean(axis=0)
    front_px = corners[:2].mean(axis=0)
    points = view.to_world([centre_px, front_px])
    if camera_height_mm:
        # The point under an overhead camera is the one seen at the image centre.
        width_px, height_px = view.image_size
        nadir = view.to_world([[width_px / 2, height_px / 2]])[0]
        points = correct_parallax(points, nadir, camera_height_mm, tag_height_mm)
    centre, front = points
    theta = math.atan2(front[1] - centre[1], front[0] - centre[0])
    # Step back from the tag centre to the middle of the wheel axis.
    offset_x, offset_y = tag_offset_mm
    x = centre[0] - (offset_x * math.cos(theta) - offset_y * math.sin(theta))
    y = centre[1] - (offset_x * math.sin(theta) + offset_y * math.cos(theta))
    return Pose(float(x), float(y), theta)


def save_tags(folder="tags", side_px=600):
    """Writes the five tags as pictures to print, each with its white border and a label."""
    os.makedirs(folder, exist_ok=True)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config.ARUCO_DICTIONARY))
    places = ["bottom-left (origin)", "bottom-right", "top-right", "top-left"]
    labels = {config.ROBOT_TAG_ID: ("robot", "ROBOT: this edge is the front")}
    for tag_id, place in zip(config.CORNER_TAG_IDS, places):
        labels[tag_id] = (f"corner_{tag_id}", f"CORNER {tag_id}: {place}")
    paths = []
    for tag_id, (name, label) in labels.items():
        border = side_px // 6
        tag = cv2.aruco.generateImageMarker(dictionary, tag_id, side_px)
        page = cv2.copyMakeBorder(tag, border + 60, border, border, border,
                                  cv2.BORDER_CONSTANT, value=255)
        cv2.putText(page, f"id {tag_id}  {label}", (border, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2, cv2.LINE_AA)
        paths.append(os.path.join(folder, f"tag_{tag_id}_{name}.png"))
        cv2.imwrite(paths[-1], page)
    return paths


def correct_parallax(points_mm, nadir_mm, camera_height_mm, tag_height_mm):
    """Moves points seen at tag height back to where they stand on the floor.

    A point at height h, seen from a camera at height H, appears farther from the point
    under the camera (the nadir) by the factor H / (H - h). This undoes it.
    """
    points = np.asarray(points_mm, dtype=np.float64).reshape(-1, 2)
    nadir = np.asarray(nadir_mm, dtype=np.float64).reshape(1, 2)
    return nadir + (points - nadir) * (camera_height_mm - tag_height_mm) / camera_height_mm


def save_print_sheet(path="tags/tags_a4.pdf", tag_mm=60.0, dpi=300):
    """Writes one A4 page with the five tags at their real size, ready to print at 100 %.

    tag_mm is the side of the black square; each tag keeps a white margin of tag_mm / 6
    inside its cut line. A 100 mm bar on the page lets you check the print scale.
    """
    px = dpi / 25.4  # pixels per mm
    page = np.full((round(297 * px), round(210 * px)), 255, dtype=np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config.ARUCO_DICTIONARY))
    margin_mm = tag_mm / 6
    cell_mm = tag_mm + 2 * margin_mm

    def text(words, x_mm, y_mm, scale=1.1):
        cv2.putText(page, words, (round(x_mm * px), round(y_mm * px)),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, 0, 2, cv2.LINE_AA)

    places = ["bottom-left (origin)", "bottom-right", "top-right", "top-left"]
    labels = [(config.ROBOT_TAG_ID, "ROBOT - this edge is the front")]
    labels += [(tag_id, f"CORNER - {place}")
               for tag_id, place in zip(config.CORNER_TAG_IDS, places)]
    columns = [(210 - 2 * cell_mm) / 3, 2 * (210 - 2 * cell_mm) / 3 + cell_mm]
    pitch = cell_mm + 12
    for index, (tag_id, label) in enumerate(labels):
        x_mm, y_mm = columns[index % 2], 16 + (index // 2) * pitch
        text(f"id {tag_id}  {label}", x_mm, y_mm - 3)
        left, top = round(x_mm * px), round(y_mm * px)
        cell_px = round(cell_mm * px)
        cv2.rectangle(page, (left, top), (left + cell_px, top + cell_px), 170, 2)  # cut line
        side_px = round(tag_mm * px)
        tag = cv2.aruco.generateImageMarker(dictionary, tag_id, side_px)
        offset = round(margin_mm * px)
        page[top + offset:top + offset + side_px, left + offset:left + offset + side_px] = tag

    # Scale check and instructions, in the free sixth slot.
    x_mm, y_mm = columns[1], 16 + 2 * pitch
    start, end = round(x_mm * px), round((x_mm + 80) * px)
    bar_y = round((y_mm + 12) * px)
    cv2.line(page, (start, bar_y), (end, bar_y), 0, 3)
    for tick in (start, end):
        cv2.line(page, (tick, bar_y - 25), (tick, bar_y + 25), 0, 3)
    text("This bar must measure 80 mm.", x_mm, y_mm + 22)
    text("Print at 100 % (actual size),", x_mm, y_mm + 32)
    text("not 'fit to page'.", x_mm, y_mm + 39)
    text("Cut along the grey lines:", x_mm, y_mm + 51)
    text("keep the white margin.", x_mm, y_mm + 58)
    text(f"Black square: {tag_mm:.0f} mm.", x_mm, y_mm + 70)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    _write_pdf(path, page, dpi)
    return path


def _write_pdf(path, grey_page, dpi):
    """Minimal one-page PDF holding a greyscale image at its true physical size."""
    height, width = grey_page.shape
    points = 72.0 / dpi  # PDF units (1/72 inch) per pixel
    page_w, page_h = width * points, height * points
    image = zlib.compress(grey_page.tobytes(), 9)
    content = f"q {page_w:.3f} 0 0 {page_h:.3f} 0 0 cm /Im0 Do Q".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_w:.3f} {page_h:.3f}] "
         f"/Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>").encode(),
        (f"<< /Type /XObject /Subtype /Image /Width {width} /Height {height} "
         f"/ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /FlateDecode "
         f"/Length {len(image)} >>\nstream\n").encode() + image + b"\nendstream",
        f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n").encode()
    with open(path, "wb") as handle:
        handle.write(out)


if __name__ == "__main__":
    for saved in save_tags() + [save_print_sheet()]:
        print("saved", saved)
