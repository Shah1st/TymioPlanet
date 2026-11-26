# Setup on a new laptop

Every command below is typed in a terminal opened **inside the `TymioPlanet` folder**.
Nothing has to be "activated": the commands call the project's own Python directly.

## 1. Install once

- **Python** 3.10 or newer, from https://www.python.org/downloads/.
  On Windows, tick "Add python.exe to PATH" in the installer.
- **Git**, from https://git-scm.com/downloads.
- **Thymio Suite**, from https://www.thymio.org/program/ (only needed to drive the robot).

Close and reopen the terminal after installing, then check:

```
python --version
git --version
```

If `python` is not found on Windows, try `py --version` and use `py` instead of `python` in
step 2. On macOS or Linux use `python3`.

## 2. Get the code

```
git clone https://github.com/Shah1st/TymioPlanet.git
cd TymioPlanet
git checkout robot-link
```

`git checkout robot-link` is needed only until that branch is merged into `main`. After the
merge, use `main`: `git checkout main`, then `git pull`.

Create the project's Python environment and install the libraries:

**Windows**
```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

**macOS / Linux**
```
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The first line creates a folder named `.venv`. If a later command says `.venv\Scripts\...` is
not recognised, that folder does not exist: you are in the wrong folder, or the first line
failed. Run `dir` (Windows) or `ls -a` (macOS / Linux) and look for `.venv`.

## 3. Run

From here on, `PY` stands for `.venv\Scripts\python` on Windows and `.venv/bin/python` on
macOS / Linux. Type the real path, not `PY`.

| Step | Command | What you should see |
|---|---|---|
| Tests | `PY -m pytest` | `45 passed` |
| Demo, no hardware | `PY -m tymioplanet.main --source synthetic` | a window with the field, the grid and an orange route |
| Robot connection | `PY -m tymioplanet.control` | three lines of sensor values, then `robot link ok`; the robot does not move |
| Camera alone | `PY -m tymioplanet.main --source 0` | the camera image with the field outlined, the grid and the route |
| Drive the route | `PY -m tymioplanet.main --source 0 --robot` | the Thymio follows the route |

For example, on Windows the demo is:

```
.venv\Scripts\python -m tymioplanet.main --source synthetic
```

Keys in the window: `q` quit (the wheels stop), `m` retake the map.

## 4. Before the robot steps

1. Plug in the wireless dongle or the USB cable and switch the Thymio on.
2. Open Thymio Suite and leave it open. Do not open the robot in one of its editors.
3. For driving: the camera is above the field and sees all four corner tags and the robot tag.

## If something goes wrong

| Message | Cause |
|---|---|
| `.venv\Scripts\... is not recognised` | the `.venv` folder is missing: redo step 2 inside the `TymioPlanet` folder |
| `python is not recognised` | Python is not installed or not on PATH: reinstall with "Add python.exe to PATH", or use `py` |
| `No module named tymioplanet` | the terminal is not in the `TymioPlanet` folder |
| `Thymio Device Manager is not running` | Thymio Suite is closed |
| `no Thymio found` | the robot is off, asleep, or not paired with the dongle |
| the window shows the laptop's own webcam | use `--source 1` instead of `--source 0` |
| `map: waiting for the 4 corner tags` | one of the corner tags is hidden or outside the picture |
