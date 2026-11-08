from __future__ import annotations

import typing
from collections.abc import Sequence

import numpy as np


def unit(v: np.ndarray | Sequence[float]) -> np.ndarray:
    """`v` scaled to length 1, zeros for a zero vector."""
    a = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(a))
    return a / n if n > 1e-12 else np.zeros_like(a)


def axis_rotation(axis: np.ndarray | Sequence[float], angle: float | np.ndarray) -> np.ndarray:
    """Rotation matrix of `angle` rad about `axis` (any length), (..., 3, 3) for an array of angles, identity about a zero axis."""
    a = unit(axis)
    k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    theta = np.asarray(angle, dtype=float)[..., None, None]
    return np.eye(3) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)


def swing(direction: np.ndarray | Sequence[float], target: np.ndarray | Sequence[float]) -> np.ndarray:
    """Smallest rotation turning `direction` onto `target`, a half turn for opposite ones, identity when either has no length."""
    a, b = unit(direction), unit(target)
    axis = np.cross(a, b)
    s, c = float(np.linalg.norm(axis)), float(a @ b)
    if s < 1e-9:
        if c >= 0.0:
            return np.eye(3)
        return axis_rotation(np.cross(a, (1.0, 0.0, 0.0) if abs(a[0]) < 0.9 else (0.0, 1.0, 0.0)), np.pi)
    return axis_rotation(axis, float(np.arctan2(s, c)))


def nearest_rotation(linear: np.ndarray) -> np.ndarray:
    """The proper rotation nearest to a matrix's top-left 3x3, scaled or mirrored ones included."""
    u, _, vt = np.linalg.svd(np.asarray(linear, dtype=float)[:3, :3])
    if np.linalg.det(u @ vt) < 0.0:
        u[:, -1] *= -1.0
    return u @ vt


class BoundingBox(tuple[tuple[float, float], tuple[float, float], tuple[float, float]]):
    def __new__(cls, sides: typing.Iterable[typing.Iterable[float]]) -> typing.Self:
        pairs = tuple((low, high) for low, high in sides)
        if len(pairs) != 3:
            raise ValueError(f"BoundingBox needs exactly 3 sides, got {len(pairs)}")
        return super().__new__(cls, pairs)

    @property
    def min_x(self) -> float:
        return self[0][0]

    @property
    def max_x(self) -> float:
        return self[0][1]

    @property
    def min_y(self) -> float:
        return self[1][0]

    @property
    def max_y(self) -> float:
        return self[1][1]

    @property
    def min_z(self) -> float:
        return self[2][0]

    @property
    def max_z(self) -> float:
        return self[2][1]

    @property
    def volume(self) -> float:
        return (self.max_x - self.min_x) * (self.max_y - self.min_y) * (self.max_z - self.min_z)

    @classmethod
    def empty(cls) -> BoundingBox:
        return cls(((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)))

    def round(self, ndigits: int = 4) -> BoundingBox:
        return BoundingBox(
            (
                (round(self.min_x, ndigits), round(self.max_x, ndigits)),
                (round(self.min_y, ndigits), round(self.max_y, ndigits)),
                (round(self.min_z, ndigits), round(self.max_z, ndigits)),
            )
        )
