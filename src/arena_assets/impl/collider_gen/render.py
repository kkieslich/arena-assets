"""Median-depth images of the Gaussians, splatted front to back like gsplat's classic mode, in plain torch."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import attrs
import numpy as np

from arena_assets.impl.collider_gen.cameras import NEAR, Camera
from arena_assets.impl.collider_gen.gaussians import GaussianSet

if TYPE_CHECKING:
    import torch

BLUR = 0.3
"""Square pixels added to every footprint's covariance."""
MIN_ALPHA = 1 / 255
MAX_ALPHA = 0.999
CHUNK = 1 << 24
"""Candidate pixels per batch of Gaussians, bounds memory."""
FIXED_POINT = 1 << 24
"""Absorption is summed as integers, so the sum is exact and deterministic on the GPU."""


@attrs.frozen
class View:
    camera: Camera
    median: np.ndarray
    """Depth in meters along the forward axis where half the light is blocked, 0 = never half."""


@attrs.frozen
class _Splats:
    """The Gaussians in front of one camera, near to far, as footprints on its image."""

    depth: torch.Tensor
    center: torch.Tensor
    """Pixel coordinates of the footprint centers."""
    conic: torch.Tensor
    """Inverse footprint covariances as (xx, xy, yy)."""
    opacity: torch.Tensor
    first: torch.Tensor
    """First pixel column and row a footprint covers."""
    extent: torch.Tensor
    """Pixel columns and rows a footprint covers."""


def render(gaussians: GaussianSet, cameras: list[Camera]) -> list[View]:
    """One view per camera."""
    import torch

    means, covariances, opacities = (torch.as_tensor(array, dtype=torch.float32, device=device()) for array in (gaussians.means, gaussians.covariances, gaussians.opacities))
    return [View(camera, median_depth(means, covariances, opacities, camera).cpu().numpy()) for camera in cameras]


def device() -> torch.device:
    """The GPU when torch sees one, else the CPU."""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def median_depth(means: torch.Tensor, covariances: torch.Tensor, opacities: torch.Tensor, camera: Camera) -> torch.Tensor:
    """Per pixel, the depth where half the light is blocked, 0 where it never is."""
    import torch

    splats = _project(means, covariances, opacities, camera)
    if len(splats.depth) == 0:
        return torch.zeros((camera.size, camera.size), dtype=means.dtype, device=means.device)
    return _first_crossing(*_fragments(splats, camera.size), camera.size)


def _project(means: torch.Tensor, covariances: torch.Tensor, opacities: torch.Tensor, camera: Camera) -> _Splats:
    import torch

    rotation = torch.as_tensor(camera.rotation, dtype=means.dtype, device=means.device)
    local = (means - torch.as_tensor(camera.position, dtype=means.dtype, device=means.device)) @ rotation.T
    keep = (local[:, 2] > NEAR) & (opacities >= MIN_ALPHA)
    local, opacities = local[keep], opacities[keep]
    covariances = rotation @ covariances[keep] @ rotation.T
    order = torch.argsort(local[:, 2], stable=True)
    local, covariances, opacities = local[order], covariances[order], opacities[order]

    x, y, z = local.unbind(1)
    focal, middle = camera.focal, camera.size / 2
    limit = 1.3 * middle / focal  # gsplat's clamp, keeps the footprints of Gaussians beside the image bounded
    jacobian = torch.zeros((len(z), 2, 3), dtype=means.dtype, device=means.device)
    jacobian[:, 0, 0] = jacobian[:, 1, 1] = focal / z
    jacobian[:, 0, 2] = -focal * (x / z).clamp(-limit, limit) / z
    jacobian[:, 1, 2] = -focal * (y / z).clamp(-limit, limit) / z
    footprint = jacobian @ covariances @ jacobian.transpose(1, 2) + BLUR * torch.eye(2, dtype=means.dtype, device=means.device)
    xx, xy, yy = footprint[:, 0, 0], footprint[:, 0, 1], footprint[:, 1, 1]
    center = torch.stack([focal * x / z + middle, focal * y / z + middle], 1)

    reach = torch.sqrt(2 * torch.log(opacities / MIN_ALPHA))[:, None] * torch.sqrt(torch.stack([xx, yy], 1))
    first = torch.ceil(center - reach - 0.5).clamp(min=0).long()
    last = torch.floor(center + reach - 0.5).clamp(max=camera.size - 1).long()
    return _Splats(
        depth=z,
        center=center,
        conic=torch.stack([yy, -xy, xx], 1) / (xx * yy - xy * xy)[:, None],
        opacity=opacities,
        first=first,
        extent=(last - first + 1).clamp(min=0),
    )


def _fragments(splats: _Splats, size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pixel index, depth and absorption wherever a footprint is opaque enough, in batches of at most CHUNK candidate pixels."""
    import torch

    device = splats.depth.device
    count = splats.extent[:, 0] * splats.extent[:, 1]
    ends = torch.cumsum(count, 0)
    begins = ends - count
    pixels, depths, absorptions = [], [], []
    start = 0
    while start < len(count):
        stop = max(start + 1, int(torch.searchsorted(ends, begins[start] + CHUNK, right=True)))
        index = torch.repeat_interleave(torch.arange(start, stop, device=device), count[start:stop])
        offset = torch.arange(len(index), device=device) + begins[start] - begins[index]
        column = splats.first[index, 0] + offset % splats.extent[index, 0]
        row = splats.first[index, 1] + offset // splats.extent[index, 0]
        dx, dy = splats.center[index, 0] - column - 0.5, splats.center[index, 1] - row - 0.5
        conic = splats.conic[index]
        falloff = 0.5 * (conic[:, 0] * dx * dx + conic[:, 2] * dy * dy) + conic[:, 1] * dx * dy
        alpha = (splats.opacity[index] * torch.exp(-falloff)).clamp(max=MAX_ALPHA)
        hit = (falloff >= 0) & (alpha >= MIN_ALPHA)
        pixels.append((row * size + column)[hit])
        depths.append(splats.depth[index][hit])
        absorptions.append(torch.round(-torch.log1p(-alpha[hit]).double() * FIXED_POINT).long())
        start = stop
    return torch.cat(pixels), torch.cat(depths), torch.cat(absorptions)


def _first_crossing(pixel: torch.Tensor, depth: torch.Tensor, absorption: torch.Tensor, size: int) -> torch.Tensor:
    """Per pixel, the depth of the first fragment at which half the light is blocked, 0 where none is."""
    import torch

    median = torch.zeros(size * size, dtype=depth.dtype, device=depth.device)
    pixel, sorted_by_pixel = torch.sort(pixel, stable=True)
    depth, absorption = depth[sorted_by_pixel], absorption[sorted_by_pixel]
    starts = torch.ones_like(pixel, dtype=torch.bool)
    starts[1:] = pixel[1:] != pixel[:-1]
    total = torch.cumsum(absorption, 0)
    start_of = torch.cummax(torch.where(starts, torch.arange(len(pixel), device=pixel.device), 0), 0).values
    blocked = total - total[start_of] + absorption[start_of] >= round(math.log(2) * FIXED_POINT)
    crossing = blocked & (starts | ~torch.roll(blocked, 1))
    median[pixel[crossing]] = depth[crossing]
    return median.reshape(size, size)
