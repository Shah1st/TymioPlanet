"""Every tunable constant of the project, in one place.

Units: millimetres, radians, seconds, unless a name says otherwise.
"""

# --- Field: the rectangle between the inner corners of the four corner tags. ---
# An A1 sheet, long side along x. The tags' inner corners must be exactly this far apart.
FIELD_WIDTH_MM = 841.0  # from tag 1 to tag 2, world x axis
FIELD_HEIGHT_MM = 594.0  # from tag 2 to tag 3, world y axis

# --- Top-down view and grid ---
MM_PER_PIXEL = 2.0  # resolution of the rectified top-down image
CELL_SIZE_MM = 10.0  # occupancy grid cell; must be a whole multiple of MM_PER_PIXEL
EDGE_MARGIN_MM = 10.0  # band along the field border that is not segmented (tag edges)

# --- Robot ---
# Thymio outline is about 110 x 110 mm (course LAB 4, src/local_occupancy.py: thymio_coords),
# so the farthest body point from its centre is the half-diagonal, 78 mm.
ROBOT_RADIUS_MM = 80.0
SAFETY_MARGIN_MM = 10.0  # added to the radius when inflating obstacles
ROBOT_MASK_RADIUS_MM = 100.0  # disc around the robot that is never segmented as an obstacle

# --- ArUco tag on top of the Thymio (its top edge must face the robot's front) ---
ARUCO_DICTIONARY = "DICT_4X4_50"
ROBOT_TAG_ID = 0
# Corner tags, placed outside the field with one corner on the field corner. Seen from the
# camera: bottom-left (world origin), bottom-right, top-right, top-left.
CORNER_TAG_IDS = (1, 2, 3, 4)
# Tag centre in the robot frame (x forward, y left), relative to the middle of the wheel axis.
TAG_OFFSET_MM = (0.0, 0.0)
# The tag sits above the floor, so away from the point under the camera it appears shifted
# outwards. Set the camera height to correct this; None leaves the pose uncorrected.
TAG_HEIGHT_MM = 55.0
CAMERA_HEIGHT_MM = None

# --- Segmentation (anything that is not floor-coloured) ---
BLUR_KERNEL = 5  # Gaussian blur before thresholding
COLOR_THRESHOLD = 30.0  # distance to the floor colour in Lab (8-bit) above which a pixel is an object
LIGHTNESS_WEIGHT = 0.5  # weight of L in that distance; below 1 so shadows count less
MORPH_KERNEL = 5  # opening then closing of each mask, in top-down pixels
MIN_BLOB_AREA_MM2 = 400.0  # smaller blobs are noise

# --- Classification of object pixels by HSV (OpenCV: H 0-179, S and V 0-255) ---
MIN_SATURATION = 70  # below this a pixel has no reliable hue (black, grey)
MIN_VALUE = 50
GREEN_HUE = (35, 85)  # planets: targets to reach
BLUE_HUE = (90, 130)  # Earth: start and final goal
# Everything else (red, black, any other colour) is an obstacle.

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
