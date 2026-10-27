import importlib.util
from pathlib import Path

import attrs
import numpy as np
import pytest
import yaml
from pxr import Gf, Usd, UsdGeom, UsdVol, Vt

pytest.importorskip("scipy", reason="collider generation needs the [collider] extra")

from scipy.spatial.transform import Rotation
from typer.testing import CliRunner

from arena_assets.__main__ import app
from arena_assets.impl import collider as collider_impl
from arena_assets.impl.collider import ColliderError
from arena_assets.impl.collider_gen import Options, attach_points, cameras, gaussians, generate, mesh, render
from arena_assets.impl.collider_gen.grid import Domain

runner = CliRunner()

LOW, HIGH = np.array([-0.3, -0.2, 0.1]), np.array([0.3, 0.2, 0.9])
LEG_RADIUS, LEG_BOTTOM = 0.02, 0.04
AXES = np.array([0.03, 0.015, 0.002])
FAINT, FLOATERS = 20, 4
CROP = ("--width", "1.4", "--depth", "1.4", "--height", "1.2")
UNDER_THE_BOX = np.array([0.2, 0.12, 0.05])


def similarity(scale: float, axis: tuple[float, float, float], degrees: float, translation: tuple[float, float, float]) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = scale * np.asarray(Gf.Matrix3d(Gf.Rotation(Gf.Vec3d(*axis), degrees))).T
    matrix[:3, 3] = translation
    return matrix


Y_UP_TO_Z_UP = similarity(1.0, (1, 0, 0), 90.0, (0, 0, 0))
TO_OBJECT = similarity(0.25, (0, 0, 1), 20.0, (0.4, -0.2, 0.1)) @ Y_UP_TO_Z_UP


def flat(values: np.ndarray) -> str:
    return " ".join(repr(float(value)) for value in np.ravel(values))


@attrs.frozen
class Capture:
    """Gaussians in the object frame as the generator should see them, and the source stage they were written to."""

    path: Path
    faces: int
    solid: int
    means: np.ndarray
    rotations: np.ndarray
    scales: np.ndarray
    opacities: np.ndarray


def _face_gaussians(rng: np.random.Generator, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Flat Gaussians tangent to the six box faces on a jittered grid, overlapping like a trained splat: first axis along a face edge, third axis along the face normal."""
    means, rotations = [], []
    for axis in range(3):
        for side, bound in ((-1.0, LOW[axis]), (1.0, HIGH[axis])):
            ticks = [np.arange(LOW[i] + spacing / 2, HIGH[i], spacing) if i != axis else np.array([bound]) for i in range(3)]
            spots = np.stack(np.meshgrid(*ticks, indexing="ij"), axis=-1).reshape(-1, 3)
            spots += rng.uniform(-spacing / 4, spacing / 4, spots.shape) * (np.arange(3) != axis)
            spots[:, axis] = bound
            per_face = len(spots)
            normal = np.eye(3)[axis] * side
            tangent = np.eye(3)[(axis + 1) % 3]
            frame = np.stack([tangent, np.cross(normal, tangent), normal], axis=1)
            means.append(spots)
            rotations.append(np.broadcast_to(frame, (per_face, 3, 3)))
    return np.concatenate(means), np.concatenate(rotations)


def _leg_gaussians(rng: np.random.Generator, count: int) -> tuple[np.ndarray, np.ndarray]:
    """A thin leg under the box center: first axis along the leg, third axis radial."""
    angle = rng.uniform(0, 2 * np.pi, count)
    radial = np.stack([np.cos(angle), np.sin(angle), np.zeros(count)], axis=1)
    means = radial * LEG_RADIUS + np.stack([np.zeros(count), np.zeros(count), rng.uniform(LEG_BOTTOM + 0.01, LOW[2], count)], axis=1)
    rotations = np.stack([np.broadcast_to([0.0, 0.0, 1.0], (count, 3)), np.cross(radial, [0.0, 0.0, 1.0]), radial], axis=2)
    return means, rotations


@pytest.fixture(scope="module")
def capture(tmp_path_factory: pytest.TempPathFactory) -> Capture:
    rng = np.random.default_rng(3)
    face_means, face_rotations = _face_gaussians(rng, 0.02)
    leg_means, leg_rotations = _leg_gaussians(rng, 60)
    faint = rng.uniform([-0.65, -0.65, 0.05], [0.65, 0.65, 1.15], (FAINT, 3))
    floaters = np.array([[0.62, 0.6, 1.1], [-0.6, 0.62, 1.05], [0.6, -0.61, 0.08], [-0.62, -0.6, 1.12]])
    means = np.concatenate([face_means, leg_means, faint, floaters])
    rotations = np.concatenate([face_rotations, leg_rotations, np.broadcast_to(np.eye(3), (FAINT + FLOATERS, 3, 3))])
    scales = np.concatenate([np.broadcast_to(AXES, (len(face_means) + len(leg_means), 3)), np.full((FAINT, 3), 0.15), np.full((FLOATERS, 3), 0.005)])
    opacities = np.concatenate([rng.uniform(0.5, 1.0, len(face_means) + len(leg_means)), np.full(FAINT, 0.05), np.full(FLOATERS, 0.9)])

    path = tmp_path_factory.mktemp("capture") / "object.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    world.AddTranslateOp().Set(Gf.Vec3d(1.0, 2.0, 3.0))
    world.AddRotateYOp().Set(30.0)
    world.AddScaleOp().Set(Gf.Vec3f(2.0, 2.0, 2.0))
    field = UsdVol.ParticleField3DGaussianSplat.Define(stage, "/World/gauss")
    to_stage = np.asarray(UsdGeom.Xformable(field).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
    to_object = TO_OBJECT @ to_stage
    meters = abs(np.linalg.det(to_object[:3, :3])) ** (1 / 3)
    turn = to_object[:3, :3] / meters
    local = (means - to_object[:3, 3]) @ np.linalg.inv(to_object[:3, :3]).T
    # scipy's as_quat is (x, y, z, w), the layout numpy sees in a quatf array, so FromNumpy stores the real part correctly
    quaternions = Rotation.from_matrix(turn.T @ rotations).as_quat()
    field.CreatePositionsAttr(Vt.Vec3fArray.FromNumpy(local.astype(np.float32)))
    field.CreateOrientationsAttr(Vt.QuatfArray.FromNumpy(quaternions.astype(np.float32)))
    field.CreateScalesAttr(Vt.Vec3fArray.FromNumpy((scales / meters).astype(np.float32)))
    field.CreateOpacitiesAttr(Vt.FloatArray.FromNumpy(opacities.astype(np.float32)))
    stage.GetRootLayer().Save()
    return Capture(path=path, faces=len(face_means), solid=len(face_means) + len(leg_means), means=means, rotations=rotations, scales=scales, opacities=opacities)


@pytest.fixture
def bundle(capture: Capture, tmp_path: Path) -> Path:
    out = tmp_path / "chair"
    result = runner.invoke(app, ["splat", str(capture.path), str(out), "--object", "--matrix", flat(TO_OBJECT), *CROP])
    assert result.exit_code == 0, result.output
    assert f"kept {len(capture.means)} of {len(capture.means)} particles" in result.output
    return out


def snapshot(bundle: Path) -> dict[str, bytes]:
    return {str(path.relative_to(bundle)): path.read_bytes() for path in sorted(bundle.rglob("*")) if path.is_file()}


def extra_installed() -> bool:
    return all(importlib.util.find_spec(module) for module in ("fast_simplification", "scipy", "skimage", "torch"))


needs_extra = pytest.mark.skipif(not extra_installed(), reason="collider generation needs 'arena_assets[collider]'")

FAST = Options(views=24, resolution=192, inside_views=8)
"""Fewer and smaller images than the defaults, so the tests stay quick."""
COARSE = Options(views=12, resolution=96, inside_views=4, voxel=0.02)
"""For tests that check no geometry."""


def inside_mesh(points: np.ndarray, triangles: np.ndarray, query: np.ndarray) -> bool:
    """Ray parity along a slightly tilted +z ray, so the ray misses edges and vertices of the grid-aligned meshes."""
    direction = np.array([1e-3, 2e-3, 1.0])
    a, b, c = (points[triangles[:, i]] for i in range(3))
    edge1, edge2 = b - a, c - a
    p = np.cross(direction, edge2)
    det = np.einsum("ij,ij->i", edge1, p)
    ok = np.abs(det) > 1e-14
    inverse = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = query - a
    u = np.einsum("ij,ij->i", s, p) * inverse
    q = np.cross(s, edge1)
    v = (q @ direction) * inverse
    t = np.einsum("ij,ij->i", edge2, q) * inverse
    return bool((ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)).sum() % 2)


def cube() -> tuple[np.ndarray, np.ndarray]:
    """A closed 0.2 m cube standing on the floor, triangles facing outward."""
    points = np.array([[x, y, z] for x in (-0.1, 0.1) for y in (-0.1, 0.1) for z in (0.0, 0.2)])
    triangles = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return points, triangles


def test_gaussians_load_in_the_bundle_frame(capture, bundle):
    loaded = gaussians.load(bundle, "chair")
    np.testing.assert_allclose(loaded.means, capture.means, atol=2e-6)
    np.testing.assert_allclose(loaded.opacities, capture.opacities, atol=1e-6)


def test_covariances_follow_the_orientation_and_scale_of_each_gaussian(capture, bundle):
    loaded = gaussians.load(bundle, "chair")
    stretched = capture.rotations * capture.scales[:, None, :]
    np.testing.assert_allclose(loaded.covariances, stretched @ stretched.transpose(0, 2, 1), atol=1e-8)
    # the face normal is the shortest axis: a swapped quaternion order would turn it
    faces = slice(0, capture.faces)
    shortest = np.linalg.eigh(loaded.covariances[faces])[1][:, :, 0]
    np.testing.assert_allclose(np.abs(np.einsum("ij,ij->i", shortest, capture.rotations[faces, :, 2])), 1.0, atol=1e-5)


def test_rotations_read_quaternions_in_numpy_order():
    quarter = Vt.QuatfArray([Gf.Quatf(np.cos(np.pi / 4), Gf.Vec3f(0.0, 0.0, np.sin(np.pi / 4)))])
    np.testing.assert_allclose(gaussians.rotations(np.asarray(quarter, np.float64))[0] @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-7)


def test_attach_points_bakes_like_an_attached_mesh_and_remove_restores_the_files(bundle):
    before = snapshot(bundle)
    baked = attach_points(bundle, *cube(), "tsdf voxel=0.01")
    assert sorted(path.name for path in baked) == ["chair.sdf", "chair.usda"]
    annotation = yaml.safe_load((bundle / "annotation.yaml").read_text())
    assert annotation["collider"] == "chair.collider.obj"
    assert annotation["collider_source"] == "tsdf voxel=0.01"
    assert Usd.Stage.Open(str(bundle / "chair.usda")).GetPrimAtPath("/chair/collider/mesh")
    assert collider_impl.remove_collider(bundle) is not None
    assert set(snapshot(bundle)) == set(before)


def test_cube_helper_is_closed_and_faces_outward():
    points, triangles = cube()
    assert mesh.closed(triangles)
    assert mesh.signed_volume(points, triangles) == pytest.approx(0.2**3)


def test_annotation_records_every_setting():
    assert Options().describe() == "tsdf floater-neighbours=16 margin=0.01 voxel=0.01 max-triangles=20000 views=64 resolution=256 inside-views=32"


def test_splat_refuses_a_generated_collider_together_with_a_mesh(capture, tmp_path):
    mesh_file = tmp_path / "mesh.obj"
    mesh_file.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    out = tmp_path / "chair"
    result = runner.invoke(app, ["splat", str(capture.path), str(out), "--object", "--matrix", flat(TO_OBJECT), *CROP, "--generate-collider", "--collider-mesh", str(mesh_file)])
    assert result.exit_code != 0 and "exclusive" in result.output
    assert not out.exists()


def test_bundle_without_particle_fields_is_refused(tmp_path):
    other = tmp_path / "empty"
    other.mkdir()
    (other / "annotation.yaml").write_text(yaml.safe_dump({"name": "empty"}))
    with pytest.raises(ColliderError, match="ParticleField"):
        generate(other)


@pytest.mark.skipif(extra_installed(), reason="the [collider] extra is installed")
def test_generation_without_the_extra_fails_clearly_and_leaves_the_bundle_without_a_collider(capture, tmp_path):
    out = tmp_path / "chair"
    result = runner.invoke(app, ["splat", str(capture.path), str(out), "--object", "--matrix", flat(TO_OBJECT), *CROP, "--generate-collider"])
    assert result.exit_code != 0
    assert "[collider] extra" in result.output and "without a collider" in result.output
    assert "collider" not in yaml.safe_load((out / "annotation.yaml").read_text())


def test_cameras_stand_in_rings_and_on_a_dome_and_look_at_the_object():
    domain = Domain(low=np.array([-0.5, -0.5, 0.0]), high=np.array([0.5, 0.5, 1.0]), crop_bottom=0.03)
    placed = cameras.place(domain, Options(views=20))
    heights = sorted(round(float(camera.position[2]), 6) for camera in placed)
    assert len(placed) == 20
    assert heights[:5] == [0.3] * 5 and heights.count(1.0) == 5
    assert sum(camera.position[2] > 1.0 for camera in placed) == 10
    center = np.array([[0.0, 0.0, 0.5]])
    for camera in placed:
        np.testing.assert_allclose(camera.rotation @ camera.rotation.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(camera.rotation) == pytest.approx(1)
        np.testing.assert_allclose(camera.rotation[2], (center[0] - camera.position) / np.linalg.norm(center[0] - camera.position), atol=1e-12)


def test_inside_cameras_stand_in_reachable_free_space_with_clearance():
    voxel, origin = 0.025, np.array([-0.5, -0.5, -0.1])
    free = np.ones((40, 40, 48), bool)
    free[10:30, 10:30, 8:36] = False  # a closed box from z = 0.1 to 0.8 ...
    free[12:28, 12:28, 10:34] = True  # ... hollow inside: free, but no robot gets there
    domain = Domain(low=np.array([-0.45, -0.45, 0.0]), high=np.array([0.45, 0.45, 1.0]), crop_bottom=0.03)
    placed = cameras.inside(free, origin, voxel, domain, Options(inside_views=16))
    assert len(placed) == 16
    for camera in placed:
        x, y, z = camera.position
        assert z == pytest.approx(cameras.RING_HEIGHTS[0]) or z == pytest.approx(cameras.RING_HEIGHTS[1])
        if z < 0.8:
            assert not (-0.25 <= x < 0.25 and -0.25 <= y < 0.25)
        assert np.all(camera.position[:2] >= domain.low[:2]) and np.all(camera.position[:2] <= domain.high[:2])
    assert len({tuple(camera.position) for camera in placed}) == 4
    assert cameras.inside(free, origin, voxel, domain, Options(inside_views=0)) == []


@needs_extra
def test_collider_holds_the_box_keeps_the_gap_fills_the_inside_and_stands_on_the_floor(bundle):
    generated = generate(bundle, FAST)
    low, high = generated.points.min(0), generated.points.max(0)
    slack = 0.08
    assert np.all(low[:2] <= LOW[:2] + 0.02) and np.all(high >= HIGH - 0.02)
    assert np.all(low[:2] >= LOW[:2] - slack) and np.all(high <= HIGH + slack)
    assert generated.watertight
    assert inside_mesh(generated.points, generated.triangles, (LOW + HIGH) / 2)
    assert not inside_mesh(generated.points, generated.triangles, UNDER_THE_BOX)
    assert inside_mesh(generated.points, generated.triangles, np.array([0.0, 0.0, 0.07]))  # the leg, unseen inside: solid
    assert inside_mesh(generated.points, generated.triangles, np.array([0.0, 0.0, 0.01]))  # and down to the floor
    assert generated.points.min(0)[2] == 0.0
    for side, face in ((-1, LOW[0]), (1, HIGH[0])):
        assert inside_mesh(generated.points, generated.triangles, np.array([face - side * 0.005, 0.0, 0.5]))
        assert not inside_mesh(generated.points, generated.triangles, np.array([face + side * (FAST.margin + 0.02), 0.0, 0.5]))


@needs_extra
def test_floaters_are_dropped_and_do_not_become_colliders(capture, bundle):
    generated = generate(bundle, COARSE)
    assert generated.total - generated.kept >= FLOATERS
    for floater in capture.means[-FLOATERS:]:
        assert not inside_mesh(generated.points, generated.triangles, floater)


@needs_extra
def test_generate_is_deterministic_and_leaves_every_bundle_file_unchanged(bundle):
    before = snapshot(bundle)
    first, second = generate(bundle, COARSE), generate(bundle, COARSE)
    np.testing.assert_array_equal(first.points, second.points)
    np.testing.assert_array_equal(first.triangles, second.triangles)
    assert first.describe() == second.describe()
    assert snapshot(bundle) == before
    line = first.describe()
    assert line.startswith("collider: ") and f"{len(first.triangles)} triangles" in line and "watertight yes" in line


@needs_extra
def test_splat_generates_the_collider_at_build_time_and_collider_removes_it(capture, tmp_path):
    out = tmp_path / "chair"
    result = runner.invoke(app, ["splat", str(capture.path), str(out), "--object", "--matrix", flat(TO_OBJECT), *CROP, "--generate-collider"])
    assert result.exit_code == 0, result.output
    assert "collider: " in result.output
    annotation = yaml.safe_load((out / "annotation.yaml").read_text())
    assert annotation["collider"] == "chair.collider.obj"
    assert annotation["collider_source"] == Options().describe()
    assert (out / "chair.sdf" / "chair.sdf").is_file()
    assert runner.invoke(app, ["collider", str(out), "--remove"]).exit_code == 0
    assert "collider" not in yaml.safe_load((out / "annotation.yaml").read_text())
    assert not (out / "chair.collider.obj").exists()


@needs_extra
def test_median_depth_is_where_half_the_light_is_blocked():
    camera = cameras.look_at(np.array([0.0, -2.0, 0.5]), np.array([0.0, 0.0, 0.5]), 65)  # odd, so the center falls on a pixel center
    blob = gaussians.GaussianSet(means=np.array([[0.0, 0.0, 0.5]]), covariances=np.eye(3)[None] * 0.05**2, opacities=np.array([0.95]))
    (view,) = render.render(blob, [camera])
    assert view.median[32, 32] == pytest.approx(2.0, abs=0.005)
    assert view.median[0, 0] == 0
    # a faint veil in front of an opaque wall: the median stays on the wall; the veil alone never blocks half the light
    veil = gaussians.GaussianSet(means=np.array([[0.0, -0.5, 0.5], [0.0, 0.0, 0.5]]), covariances=np.eye(3)[None].repeat(2, 0) * 0.05**2, opacities=np.array([0.3, 0.99]))
    (view,) = render.render(veil, [camera])
    assert view.median[32, 32] == pytest.approx(2.0, abs=0.005)
    (faint,) = render.render(gaussians.GaussianSet(veil.means[:1], veil.covariances[:1], veil.opacities[:1]), [camera])
    assert faint.median[32, 32] == 0
