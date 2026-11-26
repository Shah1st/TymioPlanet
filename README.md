# TymioPlanet

Mobile Robotics (MICRO-452) final project. A Thymio leaves Earth (blue balloon), visits every
planet (green balloons) while avoiding asteroids (red balloons and any other object), and
returns to Earth. An overhead camera watches the field, whose corners are marked by tags.

The camera takes one snapshot to build the map. After that it only tracks the robot; objects
added later are not put on the map, the robot avoids them with its own sensors.

## Status

| Component | Module | State |
|---|---|---|
| Vision: robot pose and grid map | `field`, `markers`, `colors`, `grid`, `vision` | done, tested on synthetic images |
| Global navigation | `planning.py` | Dijkstra on the grid plus visiting order; `plan_route(world_map, pose)` is wired into `main.py` |
| Local navigation | `local_avoidance.py` | wall-following program for the robot, not yet run on it; leaving the wall and rejoining the path is missing |
| Filtering | `filtering.py` | extended Kalman filter, tested in simulation; noise values not yet measured |
| Motion control | `control.py` | path follower, tested in simulation; robot link (`Thymio`) tested on the robot |

With `--robot` the Thymio drives the planned route. Not yet tried on the real field with the
camera, and local avoidance is not connected, so the robot does not react to its sensors yet.

## Run

```
python -m venv .venv
.venv\Scripts\activate            (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt

python -m tymioplanet.main --source synthetic    no hardware needed
python -m tymioplanet.main --source 0            webcam 0, the robot is not moved
python -m tymioplanet.main --source 0 --robot    webcam 0, and the Thymio drives the route
pytest
```

If webcam 0 is the laptop's own camera, try `--source 1`.

Window keys: `q` quit, `m` retake the map. With the synthetic source, `h` covers the camera.

The window shows the camera image, the rectified snapshot, and the occupancy grid
(black = seen obstacle, grey = grown by the robot radius, arrow = robot).

## With the robot (on any laptop)

1. Install [Thymio Suite](https://www.thymio.org/program/) and Python 3.10 or newer.
2. Get the code and install it:
   ```
   git clone https://github.com/Shah1st/TymioPlanet.git
   cd TymioPlanet
   python -m venv .venv
   .venv\Scripts\activate            (Linux/macOS: source .venv/bin/activate)
   pip install -r requirements.txt
   ```
3. Plug in the wireless dongle (or the USB cable), switch the Thymio on, and open Thymio
   Suite. Keep it open: it runs the Thymio Device Manager that the code talks to. Do not
   open the robot in one of its editors, or it stays reserved for that editor.
4. Check the connection. The robot does not move; it prints its sensors three times:
   ```
   python -m tymioplanet.control
   ```
5. Check the camera alone: `python -m tymioplanet.main --source 0`. The window must show the
   field outline, the grid and the route.
6. Put the robot on the field and drive: `python -m tymioplanet.main --source 0 --robot`.
   Press `q` to stop; the wheels are stopped when the program ends.

Errors: "Thymio Device Manager is not running" means Thymio Suite is closed; "no Thymio
found" means the robot is off, asleep or not paired with the dongle.

## Tags

Five ArUco tags (dictionary `DICT_4X4_50`). Print `tags/tags_a4.pdf` at 100 % on one A4 page:
all five at 6 cm, with cut lines and a bar to check the scale. Single pictures are in `tags/`
(`python -m tymioplanet.markers` regenerates them):

| Tag id | File | Where |
|---|---|---|
| 0 | `tag_0_robot.png` | on top of the Thymio, the edge under the label towards the front |
| 1 | `tag_1_corner_1.png` | field corner, bottom-left as the camera sees it (world origin) |
| 2 | `tag_2_corner_2.png` | bottom-right |
| 3 | `tag_3_corner_3.png` | top-right |
| 4 | `tag_4_corner_4.png` | top-left |

Rules:
- Each corner tag lies **outside** the field, with one corner of its black square exactly on
  the field corner. The field is the rectangle between those four inner corners. Which way
  round a corner tag is turned does not matter.
- Keep the white margin around every black square, and print on matte paper.
- The printed size is free (the code uses only the corners); about 6 cm for the black square
  works well. The robot tag must fit on the Thymio with its margin.
- Cut the label off the robot tag if it does not fit, but mark which edge is the front.

## With the real setup

1. The field is an A1 sheet, 841 x 594 mm, long side from tag 1 to tag 2. Place the corner
   tags so that their inner corners are on the sheet corners. For another size, change
   `FIELD_WIDTH_MM`, `FIELD_HEIGHT_MM` in `tymioplanet/config.py`.
2. Fix tag 0 on the Thymio. If it is not centred on the wheel axis, set `TAG_OFFSET_MM`.
3. Put the camera straight above, with all four corner tags in view.
4. Lay out the balloons and obstacles, then start the program: the first frame showing the
   four corner tags becomes the map. Press `m` to retake it.
5. Set `CAMERA_HEIGHT_MM` to correct the shift caused by the robot tag being above the floor.
6. Tune the thresholds in `config.py` under the real lighting.

## Interface (`tymioplanet/datatypes.py`)

World frame: millimetres on the field, origin at the inner corner of tag 1, x towards tag 2,
y towards tag 4; heading in radians, counter-clockwise from +x.

```python
observation = VisionSystem().update(frame)

observation.camera_ok      # False when no tag is visible (camera covered)
observation.pose           # Pose(x_mm, y_mm, theta_rad), or None if the robot tag is not seen
world_map = observation.world_map   # built once from the snapshot, None before it
world_map.grid             # int8 [row, col]: 0 free, -1 obstacle; row 0 is y = 0
world_map.grid_inflated    # same, obstacles grown by the robot radius (shown in the view)
world_map.waypoints        # [Blob(kind, x_mm, y_mm, radius_mm)], the planets
world_map.home             # Blob, Earth
```

`tymioplanet.grid.world_to_cell(x_mm, y_mm)` and `cell_to_world(row, col)` convert between
the two. The grid uses the same values as the A* / Dijkstra solution of exercise session 5.
