"""Every tunable constant of the project, in one place.

Units: millimetres, radians, seconds, unless a name says otherwise.
"""

# --- Sheet (the white surface). Measure the real one and set these two values. ---
SHEET_WIDTH_MM = 1189.0  # long side, world x axis (A0 paper until the real surface is known)
SHEET_HEIGHT_MM = 841.0  # short side, world y axis

# --- Top-down view and grid ---
MM_PER_PIXEL = 2.0  # resolution of the rectified top-down image
CELL_SIZE_MM = 10.0  # occupancy grid cell; must be a whole multiple of MM_PER_PIXEL
EDGE_MARGIN_MM = 10.0  # band along the sheet border that is not segmented (blurry paper edge)

# --- Robot ---
# Thymio outline is about 110 x 110 mm (course LAB 4, src/local_occupancy.py: thymio_coords),
# so the farthest body point from its centre is the half-diagonal, 78 mm.
ROBOT_RADIUS_MM = 80.0
SAFETY_MARGIN_MM = 10.0  # added to the radius when inflating obstacles
ROBOT_MASK_RADIUS_MM = 100.0  # disc around the robot that is never segmented as an obstacle

# --- ArUco tag on top of the Thymio (its top edge must face the robot's front) ---
ARUCO_DICTIONARY = "DICT_4X4_50"
ROBOT_TAG_ID = 0
# Tag centre in the robot frame (x forward, y left), relative to the middle of the wheel axis.
TAG_OFFSET_MM = (0.0, 0.0)
# The tag sits above the sheet, so away from the point under the camera it appears shifted
# outwards. Set the camera height to correct this; None leaves the pose uncorrected.
TAG_HEIGHT_MM = 55.0
CAMERA_HEIGHT_MM = None

# --- Sheet detection ---
MIN_SHEET_AREA_FRACTION = 0.15  # sheet must cover at least this share of the image
MIN_SHEET_BRIGHTNESS = 90  # median grey level (0-255) inside the sheet at calibration
# The camera counts as covered when the sheet gets darker than this share of its
# brightness at calibration.
COVERED_BRIGHTNESS_RATIO = 0.6

# --- Segmentation (anything that is not sheet-coloured) ---
BLUR_KERNEL = 5  # Gaussian blur before thresholding
COLOR_THRESHOLD = 30.0  # distance to the sheet colour in Lab (8-bit) above which a pixel is an object
LIGHTNESS_WEIGHT = 0.5  # weight of L in that distance; below 1 so shadows count less
MORPH_KERNEL = 5  # opening then closing of each mask, in top-down pixels
MIN_BLOB_AREA_MM2 = 400.0  # smaller blobs are noise

# --- Classification of object pixels by HSV (OpenCV: H 0-179, S and V 0-255) ---
MIN_SATURATION = 70  # below this a pixel has no reliable hue (black, grey)
MIN_VALUE = 50
GREEN_HUE = (35, 85)  # planets: targets to reach
BLUE_HUE = (90, 130)  # Earth: start and final goal
# Everything else (red, black, any other colour) is an obstacle.

# --- Map change detection ---
MAP_CHANGE_TOLERANCE_MM = 30.0  # a blob that moved less than this is the same blob

# --- Motion (shared by the path follower and the pose filter; calibrate on the real robot) ---
WHEELBASE_MM = 94.0  # distance between the wheels
# Motor units per mm/s. 2.5 is 0.4 mm/s per unit (Thymio cheat sheet: 500 = about 20 cm/s);
# the course Kalman exercise (LAB 8) measured 0.43478 mm/s per unit, i.e. 2.3.
THYMIO_UNITS_PER_MM_S = 2.5

# --- Pose filter ---
# Standard deviation of the camera pose (x mm, y mm, heading rad). Placeholder values: measure
# them by recording the pose of a robot standing still.
CAMERA_POSE_STD = (5.0, 5.0, 0.035)
# The filter rejects camera poses that disagree too much with its estimate. After this many
# rejections in a row the camera is trusted instead and the estimate restarts from it
# (robot moved by hand, or odometry drifted while the camera was covered).
FILTER_RESET_AFTER_REJECTIONS = 5
