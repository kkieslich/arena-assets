"""Triangle mesh helpers for the generated collider."""

from __future__ import annotations

import numpy as np

Mesh = tuple[np.ndarray, np.ndarray]
"""Points (N x 3, meters) and triangles (M x 3, indices into the points)."""


def weld(points: np.ndarray, triangles: np.ndarray) -> Mesh:
    """Merge coincident points and drop the triangles that collapse."""
    points, renumber = np.unique(np.round(points, 9), axis=0, return_inverse=True)
    triangles = renumber.reshape(-1)[triangles]
    a, b, c = triangles.T
    return points, triangles[(a != b) & (b != c) & (c != a)]


def closed(triangles: np.ndarray) -> bool:
    """Whether every edge is walked as often one way as the other: watertight and consistently oriented."""
    if len(triangles) == 0:
        return False
    count = int(triangles.max()) + 1
    edges = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]).astype(np.int64)
    keys, walked = np.unique(edges[:, 0] * count + edges[:, 1], return_counts=True)
    reverse = (keys % count) * count + keys // count
    found = np.searchsorted(keys, reverse).clip(max=len(keys) - 1)
    back = np.where(keys[found] == reverse, walked[found], 0)
    return bool(np.all(back == walked))


def signed_volume(points: np.ndarray, triangles: np.ndarray) -> float:
    """Enclosed volume in cubic meters, negative when the triangles face inward."""
    a, b, c = (points[triangles[:, corner]] for corner in range(3))
    return float((a * np.cross(b, c)).sum() / 6)
