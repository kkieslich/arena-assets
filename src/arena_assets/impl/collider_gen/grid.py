"""The box a collider may fill and values on a voxel grid in it, both in the bundle frame."""

from __future__ import annotations

import attrs
import numpy as np


@attrs.frozen
class Domain:
    """The crop from annotation.yaml, extended down to the floor at z = 0."""

    low: np.ndarray
    high: np.ndarray
    crop_bottom: float

    @classmethod
    def from_annotation(cls, annotation: dict) -> Domain:
        (x_low, x_high), (y_low, y_high), (z_low, z_high) = annotation["bounding_box"]
        return cls(low=np.array([x_low, y_low, min(z_low, 0.0)]), high=np.array([x_high, y_high, z_high]), crop_bottom=z_low)


@attrs.frozen
class Grid:
    """Values on the voxel centers origin + index * voxel."""

    values: np.ndarray
    origin: np.ndarray
    voxel: float
