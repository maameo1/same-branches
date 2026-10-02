"""
Bifurcation Connectedness Score (BCS).

This is the scorer used for "Same Branches, Different Trees" (STACOM 2026).
The scoring logic is unchanged from the version that produced the paper's
results; see README.md for which variant produced which result.

Algorithm (all distances in voxels, on the native image grid):

1. Skeletonise the ground-truth (GT) and predicted masks
   (``skimage.morphology.skeletonize``, 3D).
2. Junction voxels: GT skeleton voxels with >= 3 skeleton neighbours in the
   26-neighbourhood. Touching junction voxels (26-connected) form one junction.
3. Branch segments: remove every junction voxel from the GT skeleton and label
   the rest (26-connected). A segment is attached to a junction if it touches
   the junction dilated by one voxel (3x3x3 structuring element).
4. Stub: the voxels of an attached segment whose Euclidean distance to the
   junction centroid is <= ``stub_length``. This is a ball around the
   centroid, not a path length along the skeleton.
5. Dilate the predicted skeleton ``tolerance`` times with SciPy's default
   (6-connected) structuring element, i.e. a city-block ball, and label its
   connected components (26-connected) over the whole volume.
6. Each stub gets the set of component labels its voxels touch. A junction is
   preserved iff all its stubs share at least one label; a stub touching no
   component means the junction is lost. One touching voxel is enough: there
   is no minimum coverage and no distal-endpoint requirement.
7. BCS = preserved junctions / counted junctions.

``min_stub_voxels`` controls which GT junctions are counted: a junction counts
only if at least two of its stubs have >= ``min_stub_voxels`` voxels.
``min_stub_voxels=3`` (default) gives the paper's main per-case results;
``min_stub_voxels=1`` counts every junction, as in the paper's perturbation
experiment.

Known limitation: the test only looks at the stubs. A break that leaves the
first voxels of a branch attached to the junction (e.g. a gap starting a few
voxels down a child branch) is scored as preserved. See README.md and
tests/test_synthetic.py.

Author: Maame Owusu-Ansah
"""

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize
from typing import Dict, List


class BifurcationConnectivityScore:
    """Bifurcation Connectedness Score; see the module docstring."""

    def __init__(self, tolerance: int = 3, stub_length: int = 8,
                 mode: str = 'recall', min_stub_voxels: int = 3):
        """
        Args:
            tolerance: Dilation radius (voxels) for prediction skeleton
                       to handle small alignment differences.
            stub_length: Length (voxels) of branch stubs to extract around
                         each bifurcation for the connectivity check.
            mode: 'recall' (default, original BCS — GT bifurcations preserved in pred),
                  'precision' (reverse BCS — pred bifurcations that exist in GT),
                  'f1' (harmonic mean of recall and precision).
            min_stub_voxels: A source-skeleton junction is counted only if at
                             least two of its stubs have this many voxels.
                             Applies in every mode. 3 = paper main results;
                             1 = count every junction (paper perturbations).
        """
        self.tolerance = tolerance
        self.stub_length = stub_length
        self.mode = mode
        self.min_stub_voxels = min_stub_voxels

    def extract_skeleton(self, binary_mask: np.ndarray) -> np.ndarray:
        """Extract 3D skeleton from binary segmentation.

        ``> 0`` guards against scikit-image versions whose 3D skeletonize
        returns uint8 {0, 255}; it is a no-op for the boolean output of
        scikit-image 0.25, with which the paper's values reproduce.
        """
        return skeletonize(binary_mask.astype(bool)) > 0

    def find_bifurcation_points(self, skeleton: np.ndarray) -> np.ndarray:
        """
        Find bifurcation points in skeleton (voxels with >=3 neighbors).
        Returns binary mask.
        """
        kernel = np.ones((3, 3, 3), dtype=np.uint8)
        kernel[1, 1, 1] = 0

        neighbor_count = ndimage.convolve(
            skeleton.astype(np.uint8),
            kernel,
            mode='constant',
            cval=0
        )

        return (skeleton > 0) & (neighbor_count > 2)

    def get_bifurcation_clusters(self, skeleton: np.ndarray) -> List[np.ndarray]:
        """
        Cluster adjacent bifurcation voxels into individual bifurcation regions.

        Returns list of binary masks, one per bifurcation cluster.
        """
        bif_mask = self.find_bifurcation_points(skeleton)

        if not np.any(bif_mask):
            return []

        struct = ndimage.generate_binary_structure(3, 3)
        labeled, n_clusters = ndimage.label(bif_mask, structure=struct)

        clusters = []
        for i in range(1, n_clusters + 1):
            clusters.append(labeled == i)

        return clusters

    def get_branch_stubs(
        self,
        skeleton: np.ndarray,
        bif_cluster: np.ndarray
    ) -> List[np.ndarray]:
        """
        For a single bifurcation cluster, find the branch stubs meeting there.

        Removes the bifurcation voxels from the skeleton, then finds the
        connected components adjacent to the cluster. Each component is a
        branch stub (parent or child).

        Returns list of binary masks for each stub (typically 3: 1 parent + 2 children).
        """
        struct = ndimage.generate_binary_structure(3, 3)

        # Remove ALL bifurcation points (not just this cluster) to cleanly
        # separate branches
        all_bif = self.find_bifurcation_points(skeleton)
        skel_no_bif = skeleton.copy().astype(bool)
        skel_no_bif[all_bif] = False

        # Label remaining branch segments
        labeled_branches, n_branches = ndimage.label(skel_no_bif, structure=struct)

        # Find which branch segments are adjacent to this bifurcation cluster
        # Dilate the cluster by 1 voxel to catch neighbors
        dilated_cluster = ndimage.binary_dilation(bif_cluster, structure=struct,
                                                  iterations=1)

        # Find branch labels that overlap with the dilated cluster
        adjacent_labels = set(np.unique(labeled_branches[dilated_cluster])) - {0}

        stubs = []
        for lbl in adjacent_labels:
            branch_mask = (labeled_branches == lbl)
            # Only keep the stub portion close to the bifurcation
            # (within stub_length voxels)
            stub = self._extract_stub(branch_mask, bif_cluster)
            if np.any(stub):
                stubs.append(stub)

        return stubs

    def _extract_stub(
        self,
        branch_mask: np.ndarray,
        bif_cluster: np.ndarray
    ) -> np.ndarray:
        """
        Extract the portion of a branch within stub_length voxels
        of the bifurcation cluster. This is the "endpoint" we check
        for reachability in the prediction.
        """
        # Get branch voxel coordinates
        branch_coords = np.array(np.where(branch_mask)).T
        if len(branch_coords) == 0:
            return np.zeros_like(branch_mask, dtype=bool)

        # Get bifurcation cluster centroid
        bif_coords = np.array(np.where(bif_cluster)).T
        bif_centroid = bif_coords.mean(axis=0)

        # Distance of each branch voxel from the bifurcation
        dists = np.linalg.norm(branch_coords - bif_centroid, axis=1)

        # Keep voxels within stub_length
        stub_mask = np.zeros_like(branch_mask, dtype=bool)
        close_voxels = branch_coords[dists <= self.stub_length]
        if len(close_voxels) > 0:
            stub_mask[tuple(close_voxels.T)] = True

        return stub_mask

    def is_bifurcation_preserved(
        self,
        pred_skel_dilated: np.ndarray,
        stubs: List[np.ndarray]
    ) -> bool:
        """
        Check if a bifurcation is preserved in the prediction.

        A bifurcation is preserved if ALL branch stubs meeting at it
        are mutually reachable through the prediction's skeleton.

        This means: for every pair of stubs, there exists a connected
        path through pred_skel_dilated.

        Args:
            pred_skel_dilated: Prediction skeleton dilated by tolerance.
            stubs: List of branch stub masks (typically 3).

        Returns:
            True if all stubs are mutually connected through prediction.
        """
        if len(stubs) < 2:
            # Need at least 2 branches to form a meaningful bifurcation
            return len(stubs) > 0

        # Label connected components in prediction skeleton
        struct = ndimage.generate_binary_structure(3, 3)
        labeled_pred, _ = ndimage.label(pred_skel_dilated, structure=struct)

        # For each stub, find which prediction component(s) it overlaps with
        stub_components = []
        for stub in stubs:
            overlap = labeled_pred[stub]
            component_labels = set(np.unique(overlap)) - {0}
            if not component_labels:
                # This stub has no prediction skeleton nearby → not preserved
                return False
            stub_components.append(component_labels)

        # All stubs must share at least one common component
        # (i.e., they are all mutually reachable)
        common = stub_components[0]
        for sc in stub_components[1:]:
            common = common & sc
            if not common:
                return False

        return True

    def _compute_directed(
        self,
        source_skel: np.ndarray,
        target_skel_dilated: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute directed BCS: what fraction of source bifurcations are
        preserved (all stubs connected) in the target skeleton.

        For recall mode: source=GT, target=pred
        For precision mode: source=pred, target=GT
        """
        bif_clusters = self.get_bifurcation_clusters(source_skel)

        # For precision mode, filter out noisy bifurcations with tiny stubs
        if self.min_stub_voxels > 1:
            filtered = []
            for bif_cluster in bif_clusters:
                stubs = self.get_branch_stubs(source_skel, bif_cluster)
                # Keep only if at least 2 stubs meet minimum size
                valid_stubs = [s for s in stubs if np.sum(s) >= self.min_stub_voxels]
                if len(valid_stubs) >= 2:
                    filtered.append(bif_cluster)
            bif_clusters = filtered

        n_expected = len(bif_clusters)
        if n_expected == 0:
            return {'score': 1.0, 'n_preserved': 0, 'n_expected': 0}

        n_preserved = 0
        for bif_cluster in bif_clusters:
            stubs = self.get_branch_stubs(source_skel, bif_cluster)
            if self.is_bifurcation_preserved(target_skel_dilated, stubs):
                n_preserved += 1

        return {
            'score': n_preserved / n_expected,
            'n_preserved': n_preserved,
            'n_expected': n_expected
        }

    # ---------- Cached helpers (numerically identical, ~30× faster) ----------

    def _get_stubs_for_cluster_cached(
        self,
        bif_cluster: np.ndarray,
        labeled_branches: np.ndarray,
        struct: np.ndarray,
    ) -> List[np.ndarray]:
        """Per-cluster stub extraction using a pre-computed `labeled_branches`
        array. Equivalent to `get_branch_stubs` but does not re-label per call.
        """
        dilated_cluster = ndimage.binary_dilation(
            bif_cluster, structure=struct, iterations=1
        )
        adjacent_labels = set(np.unique(labeled_branches[dilated_cluster])) - {0}
        stubs = []
        for lbl in adjacent_labels:
            branch_mask = (labeled_branches == lbl)
            stub = self._extract_stub(branch_mask, bif_cluster)
            if np.any(stub):
                stubs.append(stub)
        return stubs

    def _is_bif_preserved_cached(
        self,
        labeled_target_dilated: np.ndarray,
        stubs: List[np.ndarray],
    ) -> bool:
        """Bifurcation-preservation check using a pre-labeled target. Equivalent
        to `is_bifurcation_preserved` but skips the inner `ndimage.label`.
        """
        if len(stubs) < 2:
            return len(stubs) > 0
        stub_components = []
        for stub in stubs:
            overlap = labeled_target_dilated[stub]
            component_labels = set(np.unique(overlap)) - {0}
            if not component_labels:
                return False
            stub_components.append(component_labels)
        common = stub_components[0]
        for sc in stub_components[1:]:
            common = common & sc
            if not common:
                return False
        return True

    def _compute_directed_cached(
        self,
        source_clusters: List[np.ndarray],
        labeled_source_branches: np.ndarray,
        labeled_target_dilated: np.ndarray,
        struct: np.ndarray,
    ) -> Dict[str, float]:
        """Cached equivalent of `_compute_directed`. Filtering and counting
        match the original method exactly; only the redundant labels/finds
        are removed."""
        if self.min_stub_voxels > 1:
            filtered = []
            for bif_cluster in source_clusters:
                stubs = self._get_stubs_for_cluster_cached(
                    bif_cluster, labeled_source_branches, struct
                )
                valid_stubs = [s for s in stubs if np.sum(s) >= self.min_stub_voxels]
                if len(valid_stubs) >= 2:
                    filtered.append(bif_cluster)
            source_clusters = filtered

        n_expected = len(source_clusters)
        if n_expected == 0:
            return {'score': 1.0, 'n_preserved': 0, 'n_expected': 0}

        n_preserved = 0
        for bif_cluster in source_clusters:
            stubs = self._get_stubs_for_cluster_cached(
                bif_cluster, labeled_source_branches, struct
            )
            if self._is_bif_preserved_cached(labeled_target_dilated, stubs):
                n_preserved += 1

        return {
            'score': n_preserved / n_expected,
            'n_preserved': n_preserved,
            'n_expected': n_expected,
        }

    def _build_side_cache(self, skel: np.ndarray, struct: np.ndarray) -> Dict:
        """Pre-compute everything that compute_score's inner loops re-do."""
        bif_mask = self.find_bifurcation_points(skel)
        if np.any(bif_mask):
            labeled_bifs, n_clusters = ndimage.label(bif_mask, structure=struct)
            clusters = [labeled_bifs == i for i in range(1, n_clusters + 1)]
        else:
            clusters = []
        skel_no_bif = skel.copy().astype(bool)
        skel_no_bif[bif_mask] = False
        labeled_branches, _ = ndimage.label(skel_no_bif, structure=struct)
        skel_dilated = ndimage.binary_dilation(skel, iterations=self.tolerance)
        labeled_dilated, _ = ndimage.label(skel_dilated, structure=struct)
        return {
            'clusters': clusters,
            'labeled_branches': labeled_branches,
            'labeled_dilated': labeled_dilated,
        }

    def compute_score(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute Bifurcation Connectivity Score.

        OPTIMISED 2026-05-13: pre-computes labels/find-bif ONCE per skeleton
        rather than per-bifurcation. Numerically identical to the previous
        implementation; ~30× faster on dense vessel masks.

        Args:
            pred_mask: Binary prediction (H, W, D)
            gt_mask: Binary ground truth (H, W, D)

        Returns: dict (same schema as before).
        """
        struct = ndimage.generate_binary_structure(3, 3)
        gt_skel = self.extract_skeleton(gt_mask)
        pred_skel = self.extract_skeleton(pred_mask)

        gt_cache = self._build_side_cache(gt_skel, struct)

        if self.mode == 'recall':
            # Only need pred dilated label
            pred_skel_dilated = ndimage.binary_dilation(
                pred_skel, iterations=self.tolerance
            )
            pred_labeled_dilated, _ = ndimage.label(pred_skel_dilated, structure=struct)
            r = self._compute_directed_cached(
                gt_cache['clusters'], gt_cache['labeled_branches'],
                pred_labeled_dilated, struct,
            )
            return {
                'bcs': r['score'],
                'n_preserved': r['n_preserved'],
                'n_expected': r['n_expected'],
                'preservation_rate': r['score'],
            }

        # Precision / F1 — need both sides cached
        pred_cache = self._build_side_cache(pred_skel, struct)

        recall = self._compute_directed_cached(
            gt_cache['clusters'], gt_cache['labeled_branches'],
            pred_cache['labeled_dilated'], struct,
        )
        precision = self._compute_directed_cached(
            pred_cache['clusters'], pred_cache['labeled_branches'],
            gt_cache['labeled_dilated'], struct,
        )

        if self.mode == 'precision':
            bcs = precision['score']
        else:  # f1
            r, p = recall['score'], precision['score']
            bcs = 2 * r * p / (r + p) if (r + p) > 0 else 0.0

        return {
            'bcs': bcs,
            'bcs_recall': recall['score'],
            'bcs_precision': precision['score'],
            'n_preserved': recall['n_preserved'],
            'n_expected': recall['n_expected'],
            'n_preserved_recall': recall['n_preserved'],
            'n_expected_recall': recall['n_expected'],
            'n_preserved_precision': precision['n_preserved'],
            'n_expected_precision': precision['n_expected'],
            'preservation_rate': bcs,
        }

    def __call__(
        self,
        pred: "torch.Tensor",
        target: "torch.Tensor"
    ) -> Dict[str, float]:
        """Compute BCS for PyTorch tensors ([B,] C, H, W, D; foreground > 0.5)."""
        if pred.dim() == 5:
            pred = pred[0]
            target = target[0]

        if pred.shape[0] > 1:
            pred_mask = pred[1].cpu().numpy() > 0.5
            target_mask = target[1].cpu().numpy() > 0.5
        else:
            pred_mask = pred[0].cpu().numpy() > 0.5
            target_mask = target[0].cpu().numpy() > 0.5

        return self.compute_score(pred_mask, target_mask)
