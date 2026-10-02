# same-branches

Reference implementation of the **Bifurcation Connectedness Score (BCS)** and the
controlled perturbations from:

> M. Owusu-Ansah, K. Lee, V. Venugopal, M. Jawaid, W. Duan, J. Brown.
> *Same Branches, Different Trees: A Bifurcation Connectedness Metric for Coronary
> Artery Segmentation and FFR-CT Decision Agreement.* STACOM 2026.

BCS is the fraction of ground-truth bifurcations whose branches remain mutually
reachable in a predicted segmentation. This is the code that produced the
paper's BCS values; the tests below check it against stored paper results.

## Install

```bash
pip install "git+https://github.com/maameo1/same-branches"            # metric only
pip install "same-branches[nifti,test] @ git+https://github.com/maameo1/same-branches"
```

Requires Python ≥ 3.9, NumPy, SciPy and scikit-image ≥ 0.25. The paper's values
were produced with scikit-image 0.25.2; skeletonisation can differ between
scikit-image versions, so pin it if you need to match the paper exactly.

## Quick start

```python
import nibabel as nib
from bcs import BifurcationConnectivityScore

gt = nib.load("case00751.nii.gz").get_fdata() > 0.5
pred = nib.load("case00751_pred.nii.gz").get_fdata() > 0.5   # same grid as gt

r = BifurcationConnectivityScore(tolerance=3, stub_length=8).compute_score(pred, gt)
print(r["bcs"], r["n_preserved"], r["n_expected"])
```

Or from the command line: `python examples/score_nifti.py --pred PRED.nii.gz --gt GT.nii.gz`.

## What BCS computes

All distances are in **voxels** on the grid you pass in.

1. Skeletonise the ground-truth (GT) and predicted masks (`skimage.morphology.skeletonize`, 3D).
   No spur pruning, smoothing, minimum branch length or component filtering.
2. **Junctions:** GT skeleton voxels with ≥ 3 skeleton neighbours (26-neighbourhood).
   Touching junction voxels (26-connected) form one junction; nothing else merges nearby junctions.
3. **Branch segments:** remove every junction voxel from the GT skeleton and label the remainder
   (26-connected). A segment is attached to a junction if it touches the junction dilated by one
   voxel (3×3×3).
4. **Stubs:** for each attached segment, the voxels whose **Euclidean distance to the junction
   centroid** is ≤ `stub_length` (default 8). This is a ball around the junction restricted to
   the segment, not a path length along the skeleton. (The paper's "first l = 8 voxels of each
   outgoing branch" refers to this.)
5. **Reachability:** dilate the predicted skeleton `tolerance` times (default 3) with SciPy's
   default 6-connected structuring element (a city-block ball, not a cube or sphere), then label
   its connected components (26-connected) over the **whole volume**.
6. Each stub gets the set of component labels its voxels touch. A junction is **preserved** if all
   its stubs share at least one label. A stub that touches no component makes the junction lost.
   One touching voxel is enough: there is no minimum stub coverage and no distal-endpoint test.
7. **BCS** = preserved junctions / counted junctions.

`min_stub_voxels` decides which GT junctions are counted: a junction counts only if at least two
of its stubs have ≥ `min_stub_voxels` voxels. This removes small junctions that mostly come from
skeletonisation artefacts.

`mode="f1"` also scores the reverse direction (predicted junctions checked against the GT
skeleton); `1 − bcs_precision` is the phantom-bifurcation rate (PBR).

## Settings that reproduce the paper

| Paper result | `tolerance` | `stub_length` | `min_stub_voxels` |
|---|---|---|---|
| Per-case BCS (result tables, FFR-CT analyses) | 3 | 8 | **3** (default) |
| Perturbation experiment | 3 | 8 | **1** |

Images and labels were scored on the **native ImageCAS grid** (about 0.32–0.35 mm in-plane,
0.5 mm slices), against the original, unresampled labels. The networks were trained and run at
1.0 mm isotropic; their two-channel output was mapped back to the original grid with MONAI
`Invertd` (linear interpolation of the logits) and binarised by argmax before scoring. At native
resolution the 8-voxel stub is about 2.5–3 mm in-plane and up to 4 mm across slices, and the
3-voxel tolerance is about 1–1.5 mm. If you score at a different spacing, these physical sizes
change.

Data split: the official ImageCAS v2 split. Split-1 testing (cases 751–1000) is the test set; the
remaining 750 cases were split into 638 training and 112 validation cases by a fixed random
permutation (NumPy seed 42), the same for every model and seed.

## Known limitation

BCS only examines the stubs, i.e. the first few voxels of each branch around a GT junction. A
break that leaves the start of a branch attached to the junction is scored as **preserved**, even
if the rest of the branch is cut off. On a synthetic Y-junction (radius-3 voxel tubes,
`tests/test_synthetic.py`):

| Gap in one child branch | Result |
|---|---|
| 8 voxels long, starting at the junction | lost (detected) |
| 16 voxels long, starting 4, 6, 8 or 12 voxels down the branch | preserved (not detected) |

BCS therefore measures connectedness *at* each junction, not whether the downstream subtree stays
attached. A break midway between two junctions leaves both junctions preserved. Endpoint-based
measures (e.g. the fraction of GT endpoints reachable from the main tree) cover that case.

## Perturbations

`bcs.perturbations` contains the three operators used in the paper. They edit a **prediction**
using geometry from the GT skeleton, and the result is scored against the unmodified GT.

- **Branch severing:** pick a GT junction at random and one of its attached stubs (the last in the
  list, not necessarily a child branch). Remove the stub dilated 5 times plus the whole GT skeleton
  segment containing it (junction to next junction or endpoint) dilated 4 times. The cut starts at
  the junction and spans the full segment.
- **Focal stenosis:** around a random GT skeleton voxel that is neither a junction nor an endpoint,
  take the skeleton voxels within Euclidean distance 8 and remove the shell between their 1× and 4×
  dilations, leaving a core about one voxel in radius (connectivity preserved).
- **Distal pruning:** take the GT endpoint with the largest index on the third array axis as a
  "root" (a heuristic, not the anatomical ostium); for the three endpoints furthest from it, remove
  the skeleton voxels within distance 15, dilated 4 times.

`perturb_case(pred, gt, case_id)` replays the paper's protocol for one case: NumPy's global RNG is
seeded with `42 + case_id`, cases with Dice < 0.5 or fewer than 3 GT junctions are skipped, and the
operators run in the order above with `min_stub_voxels=1`. The paper applied this to the 250 test
predictions of the no-topology baseline of each of the three architectures (N = 750) and reports
mean |ΔBCS| / mean |ΔDice| (8.7× for severing).

## Tests

```bash
pip install "same-branches[test] @ git+https://github.com/maameo1/same-branches"
pytest tests
```

The tests document the behaviour above on a synthetic Y-junction, including the known limitation.
Before release, the package was also checked against stored paper results on a sample of ImageCAS
test cases (scikit-image 0.25.2): it reproduced the per-case junction counts for both
`min_stub_voxels` settings and the perturbation outputs (voxels removed, ΔDice, ΔBCS) exactly.

## Citation

```bibtex
@inproceedings{owusuansah2026samebranches,
  author    = {Owusu-Ansah, Maame and Lee, Kelvin and Venugopal, Vinod and
               Jawaid, Moazzam and Duan, Wenting and Brown, James},
  title     = {Same Branches, Different Trees: A Bifurcation Connectedness Metric for
               Coronary Artery Segmentation and {FFR-CT} Decision Agreement},
  booktitle = {Statistical Atlases and Computational Models of the Heart (STACOM 2026)},
  year      = {2026}
}
```

## Licence

MIT. See [LICENSE](LICENSE).
