"""Virtual cameras around and inside the object, in the computer-vision convention: x right, y down, z forward."""

from __future__ import annotations

import attrs
import numpy as np
from scipy import ndimage

from arena_assets.impl.collider_gen.grid import Domain
from arena_assets.impl.collider_gen.options import Options

FIELD_OF_VIEW = np.radians(60)
INSIDE_FIELD_OF_VIEW = np.radians(90)
RING_HEIGHTS = (0.3, 1.0)
"""Meters, low and high robot sensors."""
DOME_ELEVATIONS = np.radians([20, 75])
CLEARANCE = 0.1
"""Meters between an inside camera and anything solid."""
NEAR = 0.05
"""Meters in front of a camera below which it sees nothing."""


@attrs.frozen
class Camera:
    position: np.ndarray
    rotation: np.ndarray
    """Rows are the right, down and forward directions in the bundle frame."""
    focal: float
    """Pixels."""
    size: int
    """Image width and height in pixels, pixel (0, 0) covers [0, 1) x [0, 1)."""


def level(position: np.ndarray, forward: np.ndarray, size: int, field_of_view: float) -> Camera:
    """A camera looking along forward with the image's horizontal parallel to the floor."""
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return Camera(position=position, rotation=np.stack([right, down, forward]), focal=size / 2 / np.tan(field_of_view / 2), size=size)


def look_at(position: np.ndarray, target: np.ndarray, size: int) -> Camera:
    return level(position, target - position, size, FIELD_OF_VIEW)


def place(domain: Domain, options: Options) -> list[Camera]:
    """Half the cameras in rings at RING_HEIGHTS, half on a dome, all seeing the whole domain."""
    center = (domain.low + domain.high) / 2
    reach = np.linalg.norm(domain.high - domain.low) / 2
    distance = 1.1 * reach / np.sin(FIELD_OF_VIEW / 2)

    positions = []
    per_ring = options.views // 2 // len(RING_HEIGHTS)
    for ring, height in enumerate(RING_HEIGHTS):
        for step in range(per_ring):
            angle = 2 * np.pi * (step + ring / len(RING_HEIGHTS)) / per_ring
            positions.append(np.array([center[0] + distance * np.cos(angle), center[1] + distance * np.sin(angle), height]))

    dome = options.views - len(positions)
    golden = np.pi * (3 - np.sqrt(5))
    for index in range(dome):
        elevation = DOME_ELEVATIONS[0] + (DOME_ELEVATIONS[1] - DOME_ELEVATIONS[0]) * (index + 0.5) / dome
        azimuth = index * golden
        direction = [np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)]
        positions.append(center + distance * np.array(direction))

    return [look_at(np.asarray(position, float), center, options.resolution) for position in positions]


def inside(free: np.ndarray, origin: np.ndarray, voxel: float, domain: Domain, options: Options) -> list[Camera]:
    """Four level 90 degree cameras on each of up to inside_views // 4 spots in reachable free space, spread apart."""
    wanted = options.inside_views // 4
    if wanted == 0:
        return []
    spots = _roomy_spots(free, origin, voxel, domain)
    placed = []
    for position in _spread(spots, wanted, (domain.low + domain.high) / 2):
        for quarter in range(4):
            yaw = quarter * np.pi / 2 + np.pi / 4
            placed.append(level(position, np.array([np.cos(yaw), np.sin(yaw), 0.0]), options.resolution, INSIDE_FIELD_OF_VIEW))
    return placed


def _roomy_spots(free: np.ndarray, origin: np.ndarray, voxel: float, domain: Domain) -> np.ndarray:
    """Points at RING_HEIGHTS over the domain, in free space reachable from the grid border and CLEARANCE away from anything solid."""
    pieces = np.zeros(free.shape, np.int32)
    ndimage.label(free, output=pieces)
    border = np.unique(np.concatenate([pieces[0].ravel(), pieces[-1].ravel(), pieces[:, 0].ravel(), pieces[:, -1].ravel(), pieces[..., 0].ravel(), pieces[..., -1].ravel()]))
    reachable = np.isin(pieces, border[border > 0])
    roomy = reachable & (np.asarray(ndimage.distance_transform_edt(free)) * voxel >= CLEARANCE)
    spots = [np.zeros((0, 3))]
    for height in RING_HEIGHTS:
        layer = int(round((height - origin[2]) / voxel))
        if not 0 <= layer < free.shape[2]:
            continue
        xs, ys = np.nonzero(roomy[:, :, layer])
        on_layer = np.column_stack([origin[0] + xs * voxel, origin[1] + ys * voxel, np.full(len(xs), height)])
        spots.append(on_layer[np.all((on_layer[:, :2] >= domain.low[:2]) & (on_layer[:, :2] <= domain.high[:2]), axis=1)])
    return np.concatenate(spots)


def _spread(spots: np.ndarray, count: int, center: np.ndarray) -> np.ndarray:
    """Up to count spots: the one nearest the center, then each time the one farthest from those chosen."""
    if len(spots) == 0:
        return spots
    chosen = [int(np.argmin(np.linalg.norm(spots[:, :2] - center[:2], axis=1)))]
    distance = np.linalg.norm(spots - spots[chosen[0]], axis=1)
    while len(chosen) < min(count, len(spots)):
        chosen.append(int(np.argmax(distance)))
        distance = np.minimum(distance, np.linalg.norm(spots - spots[chosen[-1]], axis=1))
    return spots[chosen]
