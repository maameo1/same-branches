"""
Controlled perturbations used in the paper's perturbation experiment.

The three operators edit a *prediction* (not the ground truth) using geometry
taken from the GT skeleton, and the result is scored against the unmodified GT.
They are copied unchanged from the code that produced the paper's numbers,
apart from ``skeletonize(...) > 0`` (a no-op with scikit-image 0.25).

``perturb_case`` replays the paper's protocol for one case: seed NumPy's global
RNG with ``42 + case_id``, skip cases with Dice < 0.5 or < 3 GT junctions, then
apply severing, stenosis and pruning in that order (the order matters because
severing and stenosis draw from the same RNG stream). The paper used
``min_stub_voxels=1`` (every junction counted) for this experiment.
"""

from typing import Dict, Optional, Tuple

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

from .metric import BifurcationConnectivityScore


def dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.sum(a & b)
    total = np.sum(a) + np.sum(b)
    if total == 0:
        return 1.0
    return 2 * inter / total


def perturbation_branch_sever(pred, gt, bcs_metric) -> Tuple[np.ndarray, Dict]:
    """Remove one branch segment of the prediction, starting at a GT junction.

    Picks a GT junction uniformly at random and takes the last of its attached
    stubs (not guaranteed to be a child branch). Sets to background: the stub
    dilated 5 times, plus the whole GT skeleton segment containing it (junction
    to next junction or endpoint) dilated 4 times (6-connected element). The
    cut therefore starts at the junction and spans the full segment.
    """
    gt_skel = skeletonize(gt) > 0
    bif_clusters = bcs_metric.get_bifurcation_clusters(gt_skel)

    if len(bif_clusters) == 0:
        return pred, {}

    # Pick a random bifurcation
    idx = np.random.randint(len(bif_clusters))
    bif = bif_clusters[idx]
    stubs = bcs_metric.get_branch_stubs(gt_skel, bif)

    if len(stubs) < 2:
        return pred, {}

    # Find the stub that overlaps with prediction, pick one to sever
    # Dilate the stub to get the actual vessel region to remove
    stub_to_sever = stubs[-1]  # Last stub (usually a child branch)

    # Dilate stub to get the vessel volume around it
    sever_region = ndimage.binary_dilation(stub_to_sever, iterations=5)

    # Also extend further along the branch direction
    # Get the branch segment beyond the stub
    all_bif = bcs_metric.find_bifurcation_points(gt_skel)
    skel_no_bif = gt_skel.copy()
    skel_no_bif[all_bif] = False

    struct = ndimage.generate_binary_structure(3, 3)
    labeled_branches, _ = ndimage.label(skel_no_bif, structure=struct)

    # Find which branch the stub belongs to
    stub_labels = set(np.unique(labeled_branches[stub_to_sever])) - {0}
    if stub_labels:
        branch_label = list(stub_labels)[0]
        branch_mask = (labeled_branches == branch_label)
        # Dilate the full branch to get vessel volume
        branch_vessel = ndimage.binary_dilation(branch_mask, iterations=4)
        sever_region = sever_region | branch_vessel

    # Apply severing
    pred_severed = pred.copy()
    pred_severed[sever_region] = False

    voxels_removed = np.sum(pred & sever_region)

    return pred_severed, {
        'type': 'branch_sever',
        'voxels_removed': int(voxels_removed),
        'pct_removed': float(voxels_removed / max(pred.sum(), 1) * 100),
        'bifurcation_idx': idx,
    }


def perturbation_focal_stenosis(pred, gt, bcs_metric) -> Tuple[np.ndarray, Dict]:
    """Narrow the prediction around a random mid-vessel GT skeleton point.

    Takes GT skeleton voxels within Euclidean distance 8 of a random skeleton
    voxel that is neither a junction nor an endpoint, and removes the shell
    between their 1x and 4x dilations, leaving a ~1-voxel-radius core
    (connectivity preserved).
    """
    gt_skel = skeletonize(gt) > 0

    # Find a mid-vessel segment (not at bifurcations, not at endpoints)
    all_bif = bcs_metric.find_bifurcation_points(gt_skel)

    # Find endpoints
    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    kernel[1, 1, 1] = 0
    neighbor_count = ndimage.convolve(gt_skel.astype(np.uint8), kernel,
                                      mode='constant', cval=0)
    endpoints = (gt_skel > 0) & (neighbor_count == 1)

    # Get skeleton points that are not bifurcations or endpoints (mid-vessel)
    mid_vessel = gt_skel & ~all_bif & ~endpoints
    mid_coords = np.array(np.where(mid_vessel)).T

    if len(mid_coords) < 20:
        return pred, {}

    # Pick a random mid-vessel point as stenosis center
    center_idx = np.random.randint(len(mid_coords))
    center = mid_coords[center_idx]

    # Get nearby skeleton points to define the stenosis segment (~10 voxels along vessel)
    skel_coords = np.array(np.where(gt_skel)).T
    dists_to_center = np.linalg.norm(skel_coords - center, axis=1)
    stenosis_segment_mask = dists_to_center <= 8  # ~8 voxel segment length
    stenosis_skel_coords = skel_coords[stenosis_segment_mask]

    if len(stenosis_skel_coords) < 5:
        return pred, {}

    # Create stenosis by eroding the vessel in this region
    # First, identify the vessel region around the stenosis segment
    stenosis_skel = np.zeros_like(pred, dtype=bool)
    for coord in stenosis_skel_coords:
        stenosis_skel[coord[0], coord[1], coord[2]] = True

    # Dilate skeleton to approximate original vessel, then create narrowed version
    # Original vessel ~ 3-4 voxel radius, stenosis ~ 1-2 voxel radius
    original_region = ndimage.binary_dilation(stenosis_skel, iterations=4)
    narrowed_region = ndimage.binary_dilation(stenosis_skel, iterations=1)

    # The stenosis effect: remove the outer shell of the vessel in this region
    erosion_mask = original_region & ~narrowed_region

    pred_stenosed = pred.copy()
    pred_stenosed[erosion_mask] = False

    voxels_removed = np.sum(pred & erosion_mask)

    return pred_stenosed, {
        'type': 'focal_stenosis',
        'voxels_removed': int(voxels_removed),
        'pct_removed': float(voxels_removed / max(pred.sum(), 1) * 100),
        'stenosis_center': center.tolist(),
        'segment_length_voxels': len(stenosis_skel_coords),
    }


def perturbation_distal_pruning(pred, gt, bcs_metric) -> Tuple[np.ndarray, Dict]:
    """Remove the tips of the three GT endpoints furthest from a 'root'.

    The root is the GT skeleton endpoint with the largest index on the third
    array axis (a heuristic, not the anatomical ostium). For each of the three
    endpoints furthest from it (Euclidean), GT skeleton voxels within distance
    15 are dilated 4 times and removed from the prediction. Deterministic.
    """
    gt_skel = skeletonize(gt) > 0

    # Find endpoints (skeleton voxels with exactly 1 neighbor)
    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    kernel[1, 1, 1] = 0
    neighbor_count = ndimage.convolve(gt_skel.astype(np.uint8), kernel,
                                      mode='constant', cval=0)
    endpoints = (gt_skel > 0) & (neighbor_count == 1)
    endpoint_coords = np.array(np.where(endpoints)).T

    if len(endpoint_coords) < 4:
        return pred, {}

    # Identify the root (ostium) - typically the endpoint with largest z-coordinate
    z_coords = endpoint_coords[:, 2]
    root_idx = np.argmax(z_coords)
    root_coord = endpoint_coords[root_idx]

    # Compute distance from root for each endpoint
    distances_from_root = np.linalg.norm(endpoint_coords - root_coord, axis=1)

    # Get distal endpoints (excluding root), sorted by distance from root
    distal_endpoints = []
    for i, ep in enumerate(endpoint_coords):
        if i == root_idx:
            continue
        distal_endpoints.append({
            'coord': ep,
            'distance_from_root': distances_from_root[i],
        })

    if len(distal_endpoints) < 3:
        return pred, {}

    # Sort by distance from root (descending) - most distal first
    distal_endpoints.sort(key=lambda x: -x['distance_from_root'])

    # Remove just the TIPS of the 3 most distal branches
    # Only remove skeleton voxels within ~15 voxels of the endpoint
    TIP_LENGTH = 15  # voxels of skeleton to remove from each tip
    n_tips_to_remove = min(3, len(distal_endpoints))

    skel_coords = np.array(np.where(gt_skel)).T
    prune_mask = np.zeros_like(pred, dtype=bool)

    for tip_info in distal_endpoints[:n_tips_to_remove]:
        ep = tip_info['coord']
        # Find skeleton voxels near this endpoint (the tip)
        dists_to_ep = np.linalg.norm(skel_coords - ep, axis=1)
        tip_mask = dists_to_ep <= TIP_LENGTH
        tip_skel_coords = skel_coords[tip_mask]

        # Create tip skeleton and dilate to get vessel volume
        tip_skel = np.zeros_like(pred, dtype=bool)
        for coord in tip_skel_coords:
            tip_skel[coord[0], coord[1], coord[2]] = True

        tip_vessel = ndimage.binary_dilation(tip_skel, iterations=4)
        prune_mask |= tip_vessel

    pred_pruned = pred.copy()
    pred_pruned[prune_mask] = False

    voxels_removed = np.sum(pred & prune_mask)

    return pred_pruned, {
        'type': 'distal_pruning',
        'tips_removed': n_tips_to_remove,
        'tip_length_voxels': TIP_LENGTH,
        'voxels_removed': int(voxels_removed),
        'pct_removed': float(voxels_removed / max(pred.sum(), 1) * 100),
        'avg_distance_from_root': float(np.mean([t['distance_from_root'] for t in distal_endpoints[:n_tips_to_remove]])),
    }


PERTURBATIONS = [
    ("branch_sever", perturbation_branch_sever),
    ("focal_stenosis", perturbation_focal_stenosis),
    ("distal_pruning", perturbation_distal_pruning),
]


def perturb_case(pred: np.ndarray, gt: np.ndarray, case_id: int,
                 bcs_metric: Optional[BifurcationConnectivityScore] = None) -> Dict:
    """Run the paper's three perturbations on one (prediction, GT) pair.

    ``case_id`` is the integer ImageCAS case number (e.g. 751); it seeds the
    RNG exactly as in the paper. Returns original metrics and, per
    perturbation, Dice/BCS after the edit and the changes.
    """
    if bcs_metric is None:
        bcs_metric = BifurcationConnectivityScore(tolerance=3, stub_length=8,
                                                  min_stub_voxels=1)
    pred, gt = pred.astype(bool), gt.astype(bool)
    np.random.seed(42 + case_id)

    d_orig = dice(pred, gt)
    r_orig = bcs_metric.compute_score(pred, gt)
    out = {"case_id": case_id, "dice_orig": float(d_orig),
           "bcs_orig": float(r_orig["bcs"]), "n_bif": int(r_orig["n_expected"]),
           "perturbations": {}}
    if d_orig < 0.50 or r_orig["n_expected"] < 3:
        return out

    for name, fn in PERTURBATIONS:
        pred_p, info = fn(pred, gt, bcs_metric)
        if not info:
            continue
        d_p = dice(pred_p, gt)
        b_p = bcs_metric.compute_score(pred_p, gt)["bcs"]
        out["perturbations"][name] = {
            "dice_after": float(d_p), "bcs_after": float(b_p),
            "delta_dice": float(d_p - d_orig), "delta_bcs": float(b_p - r_orig["bcs"]),
            "voxels_removed": info["voxels_removed"],
        }
    return out
