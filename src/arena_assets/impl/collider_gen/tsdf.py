"""The collider where virtual cameras agree the splat is solid: a truncated signed distance function (TSDF) fused from median-depth images."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from arena_assets.impl.collider_gen import cameras, render, surface
from arena_assets.impl.collider_gen.gaussians import GaussianSet
from arena_assets.impl.collider_gen.grid import Domain, Grid
from arena_assets.impl.collider_gen.mesh import Mesh
from arena_assets.impl.collider_gen.options import Options

if TYPE_CHECKING:
    import torch

TRUNCATION_VOXELS = 3


def build(gaussians: GaussianSet, domain: Domain, options: Options) -> Mesh:
    """Render around the object, fuse, render again from the free space found, fuse all views and mesh the solid."""
    views = render.render(gaussians, cameras.place(domain, options))
    grid = fuse(views, domain, options)
    standing = cameras.inside(grid.values > 0, grid.origin, grid.voxel, domain, options)
    if standing:
        views = views + render.render(gaussians, standing)
        grid = fuse(views, domain, options)
    return surface.from_solid(grid.values < 0, grid, domain, options)


def fuse(views: list[render.View], domain: Domain, options: Options) -> Grid:
    """The TSDF averaged over the views, in units of the truncation; voxels no view judged are solid (-1)."""
    import torch

    device = render.device()

    def tensor(array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(array, dtype=torch.float64, device=device)

    voxel = options.voxel
    pad = options.margin + 2 * voxel
    low, high = domain.low - pad, domain.high + pad
    shape = tuple(int(size) for size in np.ceil((high - low) / voxel) + 1)
    centers = tensor(low) + torch.stack(torch.meshgrid(*(torch.arange(size, device=device) for size in shape), indexing="ij"), -1).reshape(-1, 3) * voxel
    truncation = TRUNCATION_VOXELS * voxel

    total = torch.zeros(len(centers), dtype=torch.float64, device=device)
    votes = torch.zeros_like(total)
    for view in views:
        camera = view.camera
        in_camera = (centers - tensor(camera.position)) @ tensor(camera.rotation).T
        ahead = torch.nonzero(in_camera[:, 2] > cameras.NEAR)[:, 0]
        column, row = torch.floor(camera.focal * in_camera[ahead, :2] / in_camera[ahead, 2:] + camera.size / 2).long().unbind(1)
        on_image = (column >= 0) & (column < camera.size) & (row >= 0) & (row < camera.size)
        voxels, column, row = ahead[on_image], column[on_image], row[on_image]
        seen_depth = tensor(view.median)[row, column]
        distance = torch.where(seen_depth > 0, seen_depth - in_camera[voxels, 2], truncation)  # a ray that hit nothing passes as empty
        judged = distance > -truncation
        total[voxels[judged]] += (distance[judged] / truncation).clamp(-1, 1)
        votes[voxels[judged]] += 1
    average = torch.where(votes > 0, total / votes.clamp(min=1), -1.0)
    return Grid(values=average.reshape(shape).cpu().numpy(), origin=low, voxel=voxel)
