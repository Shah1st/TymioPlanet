"""PLACEHOLDER (motion control owner): the Thymio link and the path follower."""

from .datatypes import Pose


class Thymio:
    """Connection to the robot (tdmclient, with the Thymio Device Manager running)."""

    def connect(self) -> None:
        raise NotImplementedError

    def read_sensors(self) -> dict:
        """{'prox_horizontal': [7 ints], 'left_speed': int, 'right_speed': int}, Thymio units."""
        raise NotImplementedError

    def set_motors(self, left: int, right: int) -> None:
        """Wheel targets in Thymio units (-500..500)."""
        raise NotImplementedError


def follow_path(path_mm: list[tuple[float, float]], pose: Pose) -> tuple[int, int]:
    """Wheel targets (left, right) that move the robot along the path from its pose."""
    raise NotImplementedError
