"""The Gaussians of a splat bundle in the bundle frame."""

from __future__ import annotations

from pathlib import Path

import attrs
import numpy as np
from pxr import Usd, UsdGeom, UsdVol
from scipy.spatial.transform import Rotation

from arena_assets.impl import collider
from arena_assets.impl.collider import ColliderError
from arena_assets.impl.splat import particle_fields


@attrs.frozen
class GaussianSet:
    means: np.ndarray
    """N x 3 centers in meters."""
    covariances: np.ndarray
    """N x 3 x 3."""
    opacities: np.ndarray
    """N values in [0, 1]."""

    def __len__(self) -> int:
        return len(self.means)

    def subset(self, keep: np.ndarray) -> GaussianSet:
        return GaussianSet(self.means[keep], self.covariances[keep], self.opacities[keep])


def rotations(quaternions: np.ndarray) -> np.ndarray:
    """Rotation matrices from (x, y, z, w) quaternions, the order numpy sees in a USD quaternion array."""
    return Rotation.from_quat(quaternions).as_matrix()


def load(bundle: Path, name: str) -> GaussianSet:
    """Every ParticleField of the wrapper 'arena-assets splat' wrote, registration applied."""
    for path in collider.usd_models(bundle, name):
        stage = Usd.Stage.Open(str(path))
        fields = [field for prim in particle_fields(stage) if (field := _read_field(prim)) is not None]
        if fields:
            return GaussianSet(
                means=np.concatenate([field.means for field in fields]),
                covariances=np.concatenate([field.covariances for field in fields]),
                opacities=np.concatenate([field.opacities for field in fields]),
            )
    raise ColliderError(f"{bundle} holds no {name}.usda (or .usd, .usdc, .usdz) with ParticleField prims to build a collider from")


def _values(full: Usd.Attribute, half: Usd.Attribute) -> np.ndarray | None:
    """The full precision values, else the half precision ones, None when neither is authored."""
    for attribute in (full, half):
        value = attribute.Get() if attribute.IsValid() else None
        if value is not None and len(value):
            return np.asarray(value, np.float64)
    return None


def _read_field(prim: Usd.Prim) -> GaussianSet | None:
    position = UsdVol.ParticleFieldPositionAttributeAPI(prim)
    positions = _values(position.GetPositionsAttr(), position.GetPositionshAttr())
    if positions is None:
        return None
    count = len(positions)
    orientation = UsdVol.ParticleFieldOrientationAttributeAPI(prim)
    quaternions = _values(orientation.GetOrientationsAttr(), orientation.GetOrientationshAttr())
    scale = UsdVol.ParticleFieldScaleAttributeAPI(prim)
    scales = _values(scale.GetScalesAttr(), scale.GetScaleshAttr())
    opacity = UsdVol.ParticleFieldOpacityAttributeAPI(prim)
    opacities = _values(opacity.GetOpacitiesAttr(), opacity.GetOpacitieshAttr())

    to_bundle = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
    linear, shift = to_bundle[:3, :3], to_bundle[:3, 3]
    turn = rotations(quaternions) if quaternions is not None else np.tile(np.eye(3), (count, 1, 1))
    lengths = scales if scales is not None else np.ones((count, 3))
    sphere_to_ellipsoid = linear @ turn * lengths[:, None, :]
    return GaussianSet(
        means=positions @ linear.T + shift,
        covariances=sphere_to_ellipsoid @ sphere_to_ellipsoid.transpose(0, 2, 1),
        opacities=opacities if opacities is not None else np.ones(count),
    )
