"""Score one prediction against its ground truth (NIfTI, same grid).

    python examples/score_nifti.py --pred case00751_pred.nii.gz --gt case00751.nii.gz

Both volumes are binarised at > 0.5. Score on the ground truth's native grid:
the paper resampled predictions back to the original image grid before scoring.
"""
import argparse

import nibabel as nib

from bcs import BifurcationConnectivityScore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--min-stub-voxels", type=int, default=3,
                    help="3 = paper main results (default); 1 = count every junction")
    ap.add_argument("--tolerance", type=int, default=3)
    ap.add_argument("--stub-length", type=int, default=8)
    args = ap.parse_args()

    gt_img = nib.load(args.gt)
    gt = gt_img.get_fdata() > 0.5
    pred = nib.load(args.pred).get_fdata() > 0.5
    if pred.shape != gt.shape:
        raise SystemExit(f"shape mismatch: pred {pred.shape} vs gt {gt.shape}")

    scorer = BifurcationConnectivityScore(
        tolerance=args.tolerance, stub_length=args.stub_length,
        mode="f1", min_stub_voxels=args.min_stub_voxels)
    r = scorer.compute_score(pred, gt)
    print(f"spacing (mm): {tuple(round(float(z), 3) for z in gt_img.header.get_zooms()[:3])}")
    print(f"BCS = {r['bcs_recall']:.4f}  ({r['n_preserved_recall']}/{r['n_expected_recall']} GT junctions preserved)")
    print(f"PBR = {1 - r['bcs_precision']:.4f}  (predicted junctions without a GT counterpart)")


if __name__ == "__main__":
    main()
