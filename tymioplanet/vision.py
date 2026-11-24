"""The vision module: one snapshot makes the map, then every frame only tracks the robot."""

from . import colors, config, field, grid, markers
from .datatypes import HOME, OBSTACLE, WAYPOINT, Observation, WorldMap


class VisionSystem:
    """Holds the field calibration and the map made from the snapshot.

    update(frame) is the only call needed: it takes the snapshot on the first frame where
    the four corner tags are visible, and from then on returns the robot pose. Obstacles
    added later are not put on the map; the robot avoids those with its own sensors.
    """

    def __init__(self):
        self.view = None  # FieldView, set by the snapshot
        self.world_map = None  # the map, built once
        self.top_down = None  # rectified snapshot image, for display

    def update(self, frame):
        """Observation for one BGR frame (None counts as a covered camera)."""
        tags = markers.detect_tags(frame) if frame is not None else {}
        if self.world_map is None and tags:
            self.snap(frame, tags)
        pose = None
        if self.view is not None and config.ROBOT_TAG_ID in tags:
            pose = markers.pose_from_tag(tags[config.ROBOT_TAG_ID], self.view)
        # The corner tags never move, so a frame without any tag is a covered camera.
        return Observation(bool(tags), pose, self.world_map)

    def snap(self, frame, tags=None):
        """Locates the field and builds the map from this frame. Returns the map, or None
        when a corner tag is not visible. Call it again to retake the map."""
        if tags is None:
            tags = markers.detect_tags(frame)
        view = field.calibrate(frame, tags)
        if view is None:
            return None
        self.view = view
        self.top_down = view.warp(frame)

        valid = view.valid_mask()
        # The robot is not an obstacle: leave out a disc around it.
        ignore = None
        if config.ROBOT_TAG_ID in tags:
            pose = markers.pose_from_tag(tags[config.ROBOT_TAG_ID], view)
            centre = view.world_to_top_down([[pose.x_mm, pose.y_mm]])[0]
            ignore = colors.disc_mask(valid.shape, centre,
                                      config.ROBOT_MASK_RADIUS_MM / view.mm_per_pixel)
        masks = colors.segment(self.top_down, valid, ignore)

        factor = field.cells_factor(view.cell_size_mm, view.mm_per_pixel)
        raw = grid.mask_to_grid(masks[OBSTACLE], factor)
        homes = colors.find_blobs(masks[HOME], HOME, view)
        self.world_map = WorldMap(
            grid=raw,
            grid_inflated=grid.inflate(raw, config.ROBOT_RADIUS_MM + config.SAFETY_MARGIN_MM,
                                       view.cell_size_mm),
            cell_size_mm=view.cell_size_mm,
            obstacles=colors.find_blobs(masks[OBSTACLE], OBSTACLE, view),
            waypoints=colors.find_blobs(masks[WAYPOINT], WAYPOINT, view),
            home=homes[0] if homes else None,  # the largest blue blob
        )
        return self.world_map
