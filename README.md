# TymioPlanet

Mobile Robotics (MICRO-452) final project. A Thymio leaves Earth (blue balloon), visits every
planet (green balloons) while avoiding asteroids (red balloons and any other object), and
returns to Earth. An overhead camera watches a white sheet.

## Status

| Component | Module | State |
|---|---|---|
| Vision: robot pose and grid map | `sheet`, `markers`, `colors`, `grid`, `vision` | done, tested on synthetic images |
| Global navigation | `planning.py` | placeholder |
| Local navigation | `local_avoidance.py` | placeholder |
| Filtering | `filtering.py` | extended Kalman filter, tested in simulation; noise values not yet measured |
| Motion control | `control.py` | path follower, tested in simulation; Thymio link is a placeholder |

Each placeholder has the function signatures and a docstring with the units; replace the
`raise NotImplementedError` and `main.py` starts using it.

## Run

```
python -m venv .venv
.venv\Scripts\activate            (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt

python -m tymioplanet.main --source synthetic    no hardware needed
python -m tymioplanet.main --source 0            webcam 0
pytest
```

Window keys: `q` quit, `c` recalibrate the sheet. With the synthetic source: `h` covers the
camera, `b` adds a balloon.

The window shows the camera image, the rectified top-down view, and the occupancy grid
(black = seen obstacle, grey = grown by the robot radius, arrow = robot).

## With the real setup

1. Measure the sheet and set `SHEET_WIDTH_MM`, `SHEET_HEIGHT_MM` in `tymioplanet/config.py`.
2. `python -m tymioplanet.markers` writes `robot_tag.png`. Print it about 6 cm wide and fix it
   on the Thymio with its top edge towards the front. If it is not centred on the wheel axis,
   set `TAG_OFFSET_MM`.
3. Put the camera straight above; the whole sheet must be visible on a darker background
   when the program starts (that is when the sheet is located).
4. Set `CAMERA_HEIGHT_MM` to correct the shift caused by the tag being above the sheet.
5. Tune the thresholds in `config.py` under the real lighting.

## Interface (`tymioplanet/datatypes.py`)

World frame: millimetres on the sheet, origin at a sheet corner, x along the long side, y up;
heading in radians, counter-clockwise from +x.

```python
observation = VisionSystem().update(frame)

observation.camera_ok      # False when the camera is covered
observation.pose           # Pose(x_mm, y_mm, theta_rad), or None if the tag is not seen
observation.map_changed    # True when an object appeared, moved or vanished: re-plan
world_map = observation.world_map
world_map.grid             # int8 [row, col]: 0 free, -1 obstacle; row 0 is y = 0
world_map.grid_inflated    # same, obstacles grown by the robot radius: plan on this one
world_map.waypoints        # [Blob(kind, x_mm, y_mm, radius_mm)], the planets
world_map.home             # Blob, Earth
```

`tymioplanet.grid.world_to_cell(x_mm, y_mm)` and `cell_to_world(row, col)` convert between
the two. The grid uses the same values as the A* / Dijkstra solution of exercise session 5.
