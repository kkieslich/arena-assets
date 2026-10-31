from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from arena_assets.rig import RIG_READER, RIG_SUFFIX, Rig, RigDriver, load_rig, rig_of, write_rig

HALF = float(np.sqrt(0.5))


def _spread_rig() -> Rig:
    identity = (0.0, 0.0, 0.0, 1.0)
    return Rig(
        joints=("root", "mid", "tip"),
        parents=[-1, 0, 1],
        stance=[identity, (0.0, HALF, 0.0, HALF), identity],
        bones=("Hips", "A", "B", "Tip"),
        bone_joints=[0, 1, 1, 2],
        bone_weights=[1.0, 0.25, 0.75, 1.0],
        bone_offsets=[identity, (0.0, 0.0, HALF, HALF), identity, (HALF, 0.0, 0.0, HALF)],
        soles=[1, 2],
        drivers=(RigDriver("l_wheel", "WheelL", [0.0, 1.0, 0.0], {"from": "distance", "radius": 0.3, "lateral": 0.28}),),
        settings={"skeleton": "smplx", "root_offset": [0.0, 0.0, -0.3]},
    )


def test_a_rig_round_trips_with_spread_joints_soles_and_drivers(tmp_path: Path) -> None:
    rig = _spread_rig()
    loaded = load_rig(write_rig(tmp_path / "a.rig", rig))
    assert loaded is not None
    assert loaded.joints == ("root", "mid", "tip")
    assert loaded.parent_names == {"root": None, "mid": "root", "tip": "mid"}
    assert loaded.stance == pytest.approx(rig.stance)
    assert [(bone, weight) for bone, weight, _ in loaded.bones_of("mid")] == [("A", 0.25), ("B", 0.75)]
    assert loaded.bones_of("mid")[0][2] == pytest.approx([0.0, 0.0, HALF, HALF])
    assert loaded.sole_names == ("mid", "tip")
    (driver,) = loaded.drivers
    assert (driver.name, driver.bone, driver.axis.tolist(), driver.settings) == ("l_wheel", "WheelL", [0.0, 1.0, 0.0], {"from": "distance", "radius": 0.3, "lateral": 0.28})
    assert loaded.settings == {"skeleton": "smplx", "root_offset": [0.0, 0.0, -0.3]}
    assert loaded.version == 1


def test_a_rig_without_optional_groups_loads_with_none_of_them(tmp_path: Path) -> None:
    loaded = load_rig(write_rig(tmp_path / "a.rig", Rig(("root",), [-1], [(0.0, 0.0, 0.0, 1.0)], ("Hips",), [0], [1.0], [(0.0, 0.0, 0.0, 1.0)])))
    assert loaded is not None
    assert loaded.drivers == () and len(loaded.soles) == 0


def test_the_file_is_stored_not_compressed_and_holds_the_documented_members(tmp_path: Path) -> None:
    path = write_rig(tmp_path / "a.rig", _spread_rig())
    with zipfile.ZipFile(path) as archive:
        assert {info.compress_type for info in archive.infolist()} == {zipfile.ZIP_STORED}
        names = {info.filename.removesuffix(".npy") for info in archive.infolist()}
    assert names == {"rig/version", "rig/settings", "joints/names", "joints/parents", "joints/stance", "bones/names", "bones/joints", "bones/weights", "bones/offsets", "soles/joints", "drivers/names", "drivers/bones", "drivers/axes", "drivers/settings"}


def _rewrite(path: Path, drop: tuple[str, ...] = (), **extra: object) -> None:
    with np.load(path, allow_pickle=False) as z:
        members = {key: z[key] for key in z.files if key not in drop}
    with path.open("wb") as file:
        np.savez(file, **(members | extra))


def test_a_version_above_the_reader_is_absent_and_names_the_fix(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = write_rig(tmp_path / "a.rig", _spread_rig())
    _rewrite(path, drop=("rig/version",), **{"rig/version": np.array(RIG_READER + 1)})
    with caplog.at_level(logging.WARNING):
        assert load_rig(path) is None
    assert "update Arena" in caplog.text and str(path) in caplog.text


def test_unknown_groups_members_and_settings_keys_are_ignored(tmp_path: Path) -> None:
    path = write_rig(tmp_path / "a.rig", _spread_rig())
    _rewrite(path, **{"face/blendshapes": np.zeros((3, 2)), "joints/limits": np.zeros((3, 2)), "rig/settings": np.array(json.dumps({"skeleton": "smplx", "future_key": [1, 2]}))})
    loaded = load_rig(path)
    assert loaded is not None
    assert loaded.joints == ("root", "mid", "tip") and loaded.settings["future_key"] == [1, 2]


def test_a_file_missing_a_required_member_is_rejected_naming_it(tmp_path: Path) -> None:
    path = write_rig(tmp_path / "a.rig", _spread_rig())
    _rewrite(path, drop=("bones/offsets",))
    with pytest.raises(ValueError, match="bones/offsets"):
        load_rig(path)


def test_a_file_that_is_not_an_array_zip_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "a.rig").write_text("rig: 2\n")
    with pytest.raises(ValueError):
        load_rig(tmp_path / "a.rig")


def test_joints_listed_before_their_parent_are_rejected() -> None:
    with pytest.raises(ValueError, match="parents first"):
        Rig(["tip", "root"], [1, -1], np.tile([0.0, 0.0, 0.0, 1.0], (2, 1)), [], [], [], np.zeros((0, 4)))


def test_a_missing_file_is_absent_and_warns_once_naming_rig_human(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    sdf = tmp_path / "ghost.sdf"
    with caplog.at_level(logging.WARNING):
        assert rig_of(str(sdf)) is None
        assert rig_of(str(sdf)) is None
    assert caplog.text.count("rig_human") == 1
    assert rig_of("") is None


def test_the_rig_beside_an_sdf_shares_its_stem(tmp_path: Path) -> None:
    write_rig(tmp_path / f"alice{RIG_SUFFIX}", _spread_rig())
    rig = rig_of(str(tmp_path / "alice.sdf"))
    assert rig is not None and rig.joints == ("root", "mid", "tip")


def test_the_rig_module_loads_on_numpy_alone():
    code = "import sys, arena_assets.rig as r; assert r.Rig; print([m for m in ('arena_simulation_setup', 'attrs', 'cattrs', 'yaml') if m in sys.modules])"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)})
    assert out.stdout.strip() == "[]"
