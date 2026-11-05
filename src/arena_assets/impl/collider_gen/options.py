"""Collider generation settings."""

from __future__ import annotations

import attrs


@attrs.frozen
class Options:
    """Fixed settings; tests pass smaller ones to run faster."""

    floater_neighbours: int = 16
    """Neighbours the floater filter compares, 0 = no filter."""

    margin: float = 0.01
    """Meters the collider stands outside the surface."""

    voxel: float = 0.01
    """Grid spacing in meters."""

    max_triangles: int = 20000
    """Triangles the mesh is simplified to, more where fewer would open a hole."""

    views: int = 64
    """Cameras around the object."""

    resolution: int = 256
    """Depth image width and height in pixels."""

    inside_views: int = 32
    """Cameras in the free space after the first pass, 0 = none."""

    def describe(self) -> str:
        """The settings as annotation.yaml records them."""
        return " ".join(["tsdf", *(f"{name.replace('_', '-')}={value:g}" for name, value in attrs.asdict(self).items())])
