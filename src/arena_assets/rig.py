"""Arena .rig: a body rig as a stored numpy zip of `group/member` arrays, joints by name and parent index only."""

from __future__ import annotations

import json
import logging
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)
logger.propagate = True

RIG_SUFFIX = ".rig"
RIG_READER = 1

_REQUIRED = ("rig/version", "rig/settings", "joints/names", "joints/parents", "joints/stance", "bones/names", "bones/joints", "bones/weights", "bones/offsets")
_warned: set[str] = set()


def _names(values: object) -> tuple[str, ...]:
    return tuple(str(name) for name in values)  # pyright: ignore[reportGeneralTypeIssues]


def _quaternions(value: object) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(-1, 4)


@dataclass(frozen=True, eq=False)
class RigDriver:
    """A wire joint turned about `axis` of skin bone `bone`, `settings` holding its source (`from`, `radius`, `lateral`)."""

    name: str
    bone: str
    axis: np.ndarray
    settings: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "axis", np.asarray(self.axis, dtype=float))


@dataclass(frozen=True, eq=False)
class Rig:
    """A loaded .rig: joints parents first with their neutral stance, skin bones following joints, sole joints and drivers."""

    joints: tuple[str, ...]
    parents: np.ndarray
    stance: np.ndarray
    bones: tuple[str, ...]
    bone_joints: np.ndarray
    bone_weights: np.ndarray
    bone_offsets: np.ndarray
    soles: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    drivers: tuple[RigDriver, ...] = ()
    settings: Mapping[str, Any] = field(default_factory=dict)
    version: int = 1

    def __post_init__(self) -> None:
        converted = {
            "joints": _names(self.joints),
            "parents": np.asarray(self.parents, dtype=np.int32),
            "stance": _quaternions(self.stance),
            "bones": _names(self.bones),
            "bone_joints": np.asarray(self.bone_joints, dtype=np.int32),
            "bone_weights": np.asarray(self.bone_weights, dtype=float),
            "bone_offsets": _quaternions(self.bone_offsets),
            "soles": np.asarray(self.soles, dtype=np.int32),
        }
        for name, value in converted.items():
            object.__setattr__(self, name, value)
        joint_count, bone_count = len(self.joints), len(self.bones)
        if len(set(self.joints)) != joint_count:
            raise ValueError(f"duplicate joint names in {list(self.joints)}")
        if self.parents.shape != (joint_count,) or len(self.stance) != joint_count:
            raise ValueError(f"{joint_count} joints but {self.parents.shape[0]} parents and {len(self.stance)} stance rows")
        if any(not -1 <= parent < index for index, parent in enumerate(self.parents.tolist())):
            raise ValueError("joints must list parents first, a parent index below the joint's own or -1")
        if not (self.bone_joints.shape == self.bone_weights.shape == (bone_count,)) or len(self.bone_offsets) != bone_count:
            raise ValueError(f"{bone_count} bones but {self.bone_joints.shape[0]} joints, {self.bone_weights.shape[0]} weights and {len(self.bone_offsets)} offsets")
        if bone_count and not 0 <= int(self.bone_joints.min()) <= int(self.bone_joints.max()) < joint_count:
            raise ValueError("bone joint index out of range")
        if len(self.soles) and not 0 <= int(self.soles.min()) <= int(self.soles.max()) < joint_count:
            raise ValueError("sole joint index out of range")

    @property
    def index(self) -> dict[str, int]:
        """Joint name to index."""
        return {name: i for i, name in enumerate(self.joints)}

    @property
    def parent_names(self) -> dict[str, str | None]:
        """Joint name to parent name, None for a root."""
        return {name: (self.joints[parent] if parent >= 0 else None) for name, parent in zip(self.joints, self.parents.tolist(), strict=True)}

    @property
    def sole_names(self) -> tuple[str, ...]:
        """Names of the joints whose bones carry the sole points."""
        return tuple(self.joints[i] for i in self.soles.tolist())

    def bones_of(self, joint: str) -> list[tuple[str, float, np.ndarray]]:
        """Bone name, weight and zero-pose world rotation (xyzw) of a joint's bones in chain order."""
        i = self.index[joint]
        return [(self.bones[k], float(self.bone_weights[k]), self.bone_offsets[k]) for k in range(len(self.bones)) if self.bone_joints[k] == i]


def write_rig(path: Path, rig: Rig) -> Path:
    """Write a .rig stored, not compressed."""
    members: dict[str, object] = {
        "rig/version": np.array(rig.version),
        "rig/settings": np.array(json.dumps(dict(rig.settings))),
        "joints/names": np.array(rig.joints, dtype=str),
        "joints/parents": rig.parents,
        "joints/stance": rig.stance,
        "bones/names": np.array(rig.bones, dtype=str),
        "bones/joints": rig.bone_joints,
        "bones/weights": rig.bone_weights,
        "bones/offsets": rig.bone_offsets,
    }
    if len(rig.soles):
        members["soles/joints"] = rig.soles
    if rig.drivers:
        members |= {
            "drivers/names": np.array([d.name for d in rig.drivers], dtype=str),
            "drivers/bones": np.array([d.bone for d in rig.drivers], dtype=str),
            "drivers/axes": np.array([d.axis for d in rig.drivers], dtype=float).reshape(-1, 3),
            "drivers/settings": np.array([json.dumps(dict(d.settings)) for d in rig.drivers], dtype=str),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        np.savez(file, **members)  # pyright: ignore[reportArgumentType]
    return path


def load_rig(path: Path) -> Rig | None:
    """The .rig at `path`, None when the file is missing or needs a newer reader (warned)."""
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            missing = [name for name in _REQUIRED if name not in z.files]
            if missing:
                raise ValueError(f"{path}: not a .rig, missing {missing}")
            version = int(z["rig/version"])
            if version > RIG_READER:
                logger.warning("%s needs reader %d, this code is %d: update Arena", path, version, RIG_READER)
                return None
            drivers: tuple[RigDriver, ...] = ()
            if "drivers/names" in z.files:
                drivers = tuple(RigDriver(str(name), str(bone), axis, json.loads(str(settings))) for name, bone, axis, settings in zip(z["drivers/names"], z["drivers/bones"], z["drivers/axes"], z["drivers/settings"], strict=True))
            return Rig(
                joints=z["joints/names"].tolist(),
                parents=z["joints/parents"],
                stance=z["joints/stance"],
                bones=z["bones/names"].tolist(),
                bone_joints=z["bones/joints"],
                bone_weights=z["bones/weights"],
                bone_offsets=z["bones/offsets"],
                soles=z["soles/joints"] if "soles/joints" in z.files else np.zeros(0, dtype=np.int32),
                drivers=drivers,
                settings=json.loads(str(z["rig/settings"])),
                version=version,
            )
    except (zipfile.BadZipFile, EOFError) as e:
        raise ValueError(f"{path}: not a .rig ({e})") from e


def rig_path(model_uri: str) -> Path | None:
    """The .rig beside an actor SDF, None for an empty uri."""
    return Path(model_uri).with_suffix(RIG_SUFFIX) if model_uri else None


def rig_of(model_uri: str) -> Rig | None:
    """The .rig beside an actor SDF, None without one, which logs one warning naming rig_human per file."""
    path = rig_path(model_uri)
    if path is None:
        return None
    rig = load_rig(path)
    if rig is None and not path.is_file() and str(path) not in _warned:
        _warned.add(str(path))
        logger.warning("%s has no .rig: the ped plays its clips, run rig_human on its bundle", path)
    return rig
