"""Floater removal: Gaussians floating alone in the air."""

from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree

from arena_assets.impl.collider_gen.gaussians import GaussianSet


def drop_floaters(gaussians: GaussianSet, neighbours: int) -> GaussianSet:
    return gaussians.subset(not_floating(gaussians.means, neighbours))


def not_floating(means: np.ndarray, neighbours: int) -> np.ndarray:
    """False where the mean distance to the nearest neighbours is over 3 standard deviations above the average."""
    if neighbours == 0 or len(means) <= neighbours:
        return np.ones(len(means), bool)
    distances, _ = KDTree(means).query(means, k=neighbours + 1, workers=-1)  # the first neighbour is the point itself
    loneliness = distances[:, 1:].mean(1)
    return loneliness <= loneliness.mean() + 3 * loneliness.std()
