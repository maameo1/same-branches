"""Behaviour of BCS on a synthetic Y-junction (radius-3 voxel tubes).

These tests document what the metric does, including its known limitation:
a gap that leaves the first few voxels of a branch attached to the junction
is not detected.
"""
import numpy as np
import pytest

from bcs import BifurcationConnectivityScore, perturbation_branch_sever

N = 120
JUNCTION = np.array([50.0, 60.0, 60.0])
CHILD_END = np.array([105.0, 95.0, 60.0])


def _tube(vol, a, b, r):
    a, b = np.asarray(a, float), np.asarray(b, float)
    p = np.stack(np.indices(vol.shape), -1).astype(float)
    ab = b - a
    t = np.clip(((p - a) @ ab) / (ab @ ab), 0, 1)
    vol |= np.linalg.norm(p - (a + t[..., None] * ab), axis=-1) <= r


@pytest.fixture(scope="module")
def y_tree():
    gt = np.zeros((N, N, N), bool)
    _tube(gt, [5, 60, 60], JUNCTION, 3.5)
    _tube(gt, JUNCTION, [105, 25, 60], 3.0)
    _tube(gt, JUNCTION, CHILD_END, 3.0)
    return gt


def _with_gap(gt, start, length):
    """Remove a slab of one child branch, `start` voxels from the junction."""
    u = (CHILD_END - JUNCTION) / np.linalg.norm(CHILD_END - JUNCTION)
    rel = np.stack(np.indices(gt.shape), -1) - JUNCTION
    along = rel @ u
    off_axis = np.linalg.norm(rel - along[..., None] * u, axis=-1)
    pred = gt.copy()
    pred[(off_axis <= 4.5) & (along >= start) & (along < start + length)] = False
    return pred


@pytest.mark.parametrize("min_stub_voxels", [1, 3])
def test_identical_prediction_preserves_junction(y_tree, min_stub_voxels):
    r = BifurcationConnectivityScore(min_stub_voxels=min_stub_voxels).compute_score(y_tree, y_tree)
    assert (r["n_preserved"], r["n_expected"]) == (1, 1)


def test_gap_at_junction_is_detected(y_tree):
    r = BifurcationConnectivityScore().compute_score(_with_gap(y_tree, start=0, length=8), y_tree)
    assert r["n_preserved"] == 0


@pytest.mark.parametrize("start", [4, 6, 8, 12])
def test_known_limitation_gap_beyond_junction_not_detected(y_tree, start):
    """A 16-voxel gap starting >= 4 voxels down the branch still scores as preserved."""
    r = BifurcationConnectivityScore().compute_score(_with_gap(y_tree, start=start, length=16), y_tree)
    assert r["n_preserved"] == 1


def test_paper_severing_operator_is_detected(y_tree):
    np.random.seed(0)
    scorer = BifurcationConnectivityScore(min_stub_voxels=1)
    pred, info = perturbation_branch_sever(y_tree, y_tree, scorer)
    assert info["voxels_removed"] > 0
    assert scorer.compute_score(pred, y_tree)["n_preserved"] == 0
