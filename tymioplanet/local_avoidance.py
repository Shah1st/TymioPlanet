"""PLACEHOLDER (local navigation owner): avoid obstacles that are not on the map."""


def avoid(prox_horizontal: list[int]) -> tuple[int, int] | None:
    """Wheel targets (left, right) in Thymio units while an unmapped obstacle is close.

    prox_horizontal: the 5 front proximity readings (0 = nothing, about 4000+ = very close).
    Returns None when the way is clear, so the path follower keeps control.
    Must work with the camera covered: it uses the robot's sensors only.
    """
    raise NotImplementedError
