import math

import numpy as np
import pytest

from arena_assets.utils.geom import BoundingBox, axis_rotation, nearest_rotation, unit


def test_accessors_and_volume():
    box = BoundingBox(((-1.0, 1.0), (0.0, 2.0), (3.0, 4.0)))
    assert (box.min_x, box.max_x) == (-1.0, 1.0)
    assert (box.min_y, box.max_y) == (0.0, 2.0)
    assert (box.min_z, box.max_z) == (3.0, 4.0)
    assert box.volume == 4.0


def test_empty():
    assert BoundingBox.empty().volume == 0.0


def test_round():
    box = BoundingBox(((0.123456, 1.0), (0.0, 1.0), (0.0, 1.0))).round(2)
    assert box.min_x == 0.12


def test_unit_scales_to_length_one_and_keeps_a_zero_vector_at_zero():
    assert unit([3.0, 0.0, 4.0]) == pytest.approx([0.6, 0.0, 0.8])
    assert np.array_equal(unit(np.zeros(3)), np.zeros(3))


def test_axis_rotation_about_a_zero_axis_is_identity():
    assert np.array_equal(axis_rotation(np.zeros(3), 1.3), np.eye(3))


@pytest.mark.parametrize("axis", [(0.0, 0.0, 2.5), (1.0, -2.0, 0.5), (-3.0, 0.0, 0.0)])
def test_axis_rotation_turns_about_a_non_unit_axis(axis):
    got = axis_rotation(np.array(axis), 0.8)
    assert got @ np.array(axis) == pytest.approx(np.array(axis), abs=1e-12)
    assert got @ got.T == pytest.approx(np.eye(3), abs=1e-12)
    assert np.trace(got) == pytest.approx(1.0 + 2.0 * math.cos(0.8), abs=1e-12)
    c, s = math.cos(0.8), math.sin(0.8)
    assert axis_rotation((0.0, 0.0, 2.5), 0.8) == pytest.approx(np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]), abs=1e-12)


def test_axis_rotation_batches_over_angles():
    angles = np.array([0.0, 0.4, -1.1])
    batched = axis_rotation((1.0, 2.0, 0.5), angles)
    assert batched.shape == (3, 3, 3)
    for got, angle in zip(batched, angles, strict=True):
        assert got == pytest.approx(axis_rotation((1.0, 2.0, 0.5), float(angle)), abs=1e-12)


@pytest.mark.parametrize("seed", range(5))
def test_nearest_rotation_of_a_scaled_or_padded_rotation_is_that_rotation(seed):
    rotation = nearest_rotation(np.random.default_rng(seed).normal(size=(3, 3)))
    assert rotation @ rotation.T == pytest.approx(np.eye(3), abs=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    for matrix in (2.5 * rotation, np.pad(rotation, (0, 1))):
        assert nearest_rotation(matrix) == pytest.approx(rotation, abs=1e-12)


def test_nearest_rotation_of_a_mirror_is_a_proper_rotation():
    rotation = nearest_rotation(np.diag([-1.0, 1.0, 1.0]))
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    assert np.trace(rotation) == pytest.approx(-1.0)
