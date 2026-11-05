"""Collider generation for splat bundles from the Gaussians alone, baked like a hand-made collider."""

from __future__ import annotations

from pathlib import Path

import attrs
import numpy as np
import yaml

from arena_assets.impl import ANNOTATION_NAME, collider
from arena_assets.impl.collider import ColliderError
from arena_assets.impl.collider_gen import mesh
from arena_assets.impl.collider_gen.grid import Domain
from arena_assets.impl.collider_gen.options import Options

__all__ = ["Generated", "Options", "attach_points", "generate"]

SOURCE_FIELD = "collider_source"


@attrs.frozen(eq=False)
class Generated:
    points: np.ndarray
    triangles: np.ndarray
    total: int
    """Gaussians in the bundle."""
    kept: int
    """Gaussians left after dropping the floaters."""
    watertight: bool
    source: str
    """The settings, as annotation.yaml records them."""

    def describe(self) -> str:
        """One console line."""
        size = " x ".join(f"{value:g}" for value in np.round(self.points.max(0) - self.points.min(0), 4))
        return f"collider: {self.kept} of {self.total} Gaussians used ({self.total - self.kept} floaters), {len(self.triangles)} triangles, watertight {'yes' if self.watertight else 'no'}, {size} m"


def _annotation(bundle: Path) -> tuple[Path, dict]:
    path = bundle / ANNOTATION_NAME
    if not path.is_file():
        raise ColliderError(f"{bundle} has no {ANNOTATION_NAME}, pass the OUT_DIR 'arena-assets splat' wrote")
    annotation = yaml.safe_load(path.read_text()) or {}
    if not annotation.get("name"):
        raise ColliderError(f"{path} names no asset ('name:' is missing), which the model files and the collider are named after")
    return path, annotation


def generate(bundle: Path, options: Options | None = None) -> Generated:
    """The collider of a splat bundle in the bundle frame, without touching the bundle."""
    options = options or Options()
    _, annotation = _annotation(bundle)
    try:
        from arena_assets.impl.collider_gen import filters, gaussians, tsdf

        every = gaussians.load(bundle, annotation["name"])
        kept = filters.drop_floaters(every, options.floater_neighbours)
        points, triangles = tsdf.build(kept, Domain.from_annotation(annotation), options)
    except ImportError:
        raise ColliderError("collider generation needs the [collider] extra of arena_assets (scikit-image, fast-simplification, scipy, torch)") from None
    return Generated(points, triangles, total=len(every), kept=len(kept), watertight=mesh.closed(triangles), source=options.describe())


def attach_points(bundle: Path, points: np.ndarray, triangles: np.ndarray, source: str) -> list[Path]:
    """Bake a collider into the bundle, add the collision-only SDF a splat bundle lacks and record it in annotation.yaml."""
    annotation_path, annotation = _annotation(bundle)
    name = annotation["name"]
    if not collider.usd_models(bundle, name):
        raise ColliderError(f"{bundle} holds no {name}.usda (or .usd, .usdc, .usdz) to bake the collider into")
    if collider.sdf_model_file(bundle, name) is None:
        collider.collision_only_sdf(bundle, name)
    baked = collider.bake(bundle, name, points, triangles)
    annotation[collider.FIELD] = f"{name}{collider.SUFFIX}"
    annotation[SOURCE_FIELD] = source
    annotation_path.write_text(yaml.safe_dump(annotation))
    return baked
