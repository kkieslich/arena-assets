"""A closed collider mesh from solid and empty voxels."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from arena_assets.impl.collider import ColliderError
from arena_assets.impl.collider_gen import mesh
from arena_assets.impl.collider_gen.grid import Domain, Grid
from arena_assets.impl.collider_gen.mesh import Mesh
from arena_assets.impl.collider_gen.options import Options

FLOOR_SLACK_VOXELS = 2
"""Voxels above the crop bottom where a column still counts as standing on it."""


def from_solid(solid: np.ndarray, grid: Grid, domain: Domain, options: Options) -> Mesh:
    """Extend to the floor, fill hollows, grow by the margin, mesh and simplify."""
    if not solid.any():
        raise ColliderError("collider generation found no solid voxel, check the crop")
    solid = extend_to_floor(solid, grid, domain.crop_bottom)
    solid = np.asarray(ndimage.binary_fill_holes(solid))
    points, triangles = triangulate(signed_distance(solid, grid), options.margin, domain)
    return simplify(points, triangles, options.max_triangles, domain)


def extend_to_floor(solid: np.ndarray, grid: Grid, crop_bottom: float) -> np.ndarray:
    """Fill each column whose lowest solid voxel is on the crop bottom down to z = 0; columns under a seat stay open."""
    floor = round(-grid.origin[2] / grid.voxel)
    lowest = solid.argmax(axis=2)
    standing = solid.any(axis=2) & (grid.origin[2] + lowest * grid.voxel <= crop_bottom + FLOOR_SLACK_VOXELS * grid.voxel)
    layer = np.arange(solid.shape[2])
    return solid | (standing[..., None] & (layer >= floor) & (layer < lowest[..., None]))


def signed_distance(solid: np.ndarray, grid: Grid) -> Grid:
    """Meters to the surface, negative inside."""
    outside = np.asarray(ndimage.distance_transform_edt(~solid))
    inside = np.asarray(ndimage.distance_transform_edt(solid))
    return Grid(values=(outside - inside) * grid.voxel, origin=grid.origin, voxel=grid.voxel)


def triangulate(distance: Grid, level: float, domain: Domain) -> Mesh:
    """Marching cubes at the level, clipped to the domain, triangles facing outward."""
    from skimage.measure import marching_cubes

    corners, triangles, _, _ = marching_cubes(distance.values, level=level)
    points = np.clip(distance.origin + corners * distance.voxel, domain.low, domain.high)
    points, triangles = mesh.weld(points, triangles)
    if mesh.signed_volume(points, triangles) < 0:
        triangles = triangles[:, ::-1]
    return points, triangles


def simplify(points: np.ndarray, triangles: np.ndarray, max_triangles: int, domain: Domain) -> Mesh:
    """At most max_triangles, or up to 4 times that where fewer would open a hole, clipped to the domain."""
    if len(triangles) <= max_triangles:
        return points, triangles
    import fast_simplification

    for target in (max_triangles, 2 * max_triangles, 4 * max_triangles):
        if target >= len(triangles):
            break
        simple_points, simple_triangles = fast_simplification.simplify(points, triangles, target_count=target)[:2]
        if mesh.closed(np.asarray(simple_triangles)):
            points, triangles = np.asarray(simple_points, float), np.asarray(simple_triangles)
            break
    return np.clip(points, domain.low, domain.high), triangles
