"""
enrich_patches.py — Post-processing enrichment for existing patch output
=========================================================================
Run this AFTER your existing create_patches.py has finished.

For every sample_NNN folder found under patches_dir, it generates:
  ├── mask_100m.npy   K-Means pseudo-semantic mask (K=4, from tir_100m.npy)
  └── edge_100m.npy   Canny edge map              (from tir_100m.npy)

These are needed for:
  - SPADE colorization training  (mask_100m.npy)
  - ControlNet diffusion training (edge_100m.npy)

Usage:
    python scripts/enrich_patches.py                          # default: output/patches
    python scripts/enrich_patches.py --patches_dir my/dir    # custom path
    python scripts/enrich_patches.py --skip_masks            # edges only
    python scripts/enrich_patches.py --skip_edges            # masks only
    python scripts/enrich_patches.py --overwrite             # redo existing files

Expected input layout (produced by the original create_patches.py):
    patches_dir/
        {city}/
            sample_000/
                tir_100m.npy   ← source for mask + edge
                tir_200m.npy
                rgb_100m.npy
            sample_001/ ...

Output (added IN-PLACE to each sample folder):
    patches_dir/{city}/sample_NNN/
        mask_100m.npy   uint8  (H, W)    values {0,1,2,3}
        edge_100m.npy   uint8  (H, W)    values {0, 255}

Label semantics for mask_100m (temperature-ordered):
    0 = cold   → water / rivers / lakes
    1 = cool   → dense vegetation / forests
    2 = warm   → urban areas / roads / buildings
    3 = hot    → bare soil / desert / exposed rock
"""

import os
import glob
import argparse
import numpy as np
import cv2
from sklearn.cluster import KMeans

# ── Constants ─────────────────────────────────────────────────────────────────
N_CLUSTERS    = 4
CANNY_LOW     = 50
CANNY_HIGH    = 150


# ── Helpers ───────────────────────────────────────────────────────────────────

def percentile_stretch_u8(arr: np.ndarray, low: int = 2, high: int = 98) -> np.ndarray:
    lo  = np.percentile(arr, low)
    hi  = np.percentile(arr, high)
    if hi - lo < 1e-5:
        return np.zeros_like(arr, dtype=np.uint8)
    return np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def make_mask(tir_2d: np.ndarray, n_clusters: int = N_CLUSTERS) -> np.ndarray:
    """K-Means on raw TIR values → temperature-sorted uint8 label map."""
    H, W = tir_2d.shape
    flat = tir_2d.reshape(-1, 1).astype(np.float32)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    raw_labels = km.fit_predict(flat)

    # Re-label: 0 = coldest centroid, K-1 = hottest
    order = np.argsort(km.cluster_centers_.flatten())
    remap = np.empty(n_clusters, dtype=np.int32)
    for new_lbl, old_lbl in enumerate(order):
        remap[old_lbl] = new_lbl

    return remap[raw_labels].reshape(H, W).astype(np.uint8)


def make_edge(tir_2d: np.ndarray) -> np.ndarray:
    """Canny edge detection on percentile-stretched TIR → binary uint8 map."""
    u8    = percentile_stretch_u8(tir_2d)
    blur  = cv2.GaussianBlur(u8, (3, 3), 0)
    return cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)   # {0, 255}


# ── Main ──────────────────────────────────────────────────────────────────────

def enrich_patches(patches_dir: str,
                   skip_masks: bool = False,
                   skip_edges: bool = False,
                   overwrite:  bool = False):

    # Find all sample directories (supports both flat and sample_NNN layouts)
    sample_dirs = sorted(glob.glob(os.path.join(patches_dir, '*'))) + \
                  sorted(glob.glob(os.path.join(patches_dir, '*', 'sample_*')))
    sample_dirs = [d for d in sample_dirs if os.path.isdir(d)]

    if not sample_dirs:
        print(f"[ERROR] No directories found under: {patches_dir}")
        print("  Make sure create_patches.py has already been run.")
        return

    print(f"Found {len(sample_dirs)} samples under {patches_dir}")
    print(f"  Generate masks : {not skip_masks}")
    print(f"  Generate edges : {not skip_edges}")
    print(f"  Overwrite      : {overwrite}")
    print()

    done_masks = 0
    done_edges = 0
    skipped    = 0
    errors     = 0

    for i, sample_dir in enumerate(sample_dirs, 1):
        tir_path  = os.path.join(sample_dir, 'tir_100m.npy')
        mask_path = os.path.join(sample_dir, 'mask_100m.npy')
        edge_path = os.path.join(sample_dir, 'edge_100m.npy')

        if not os.path.exists(tir_path):
            print(f"  [{i:>5}/{len(sample_dirs)}] SKIP (no tir_100m.npy): {sample_dir}")
            skipped += 1
            continue

        try:
            tir = np.load(tir_path).astype(np.float32)
            tir_2d = tir[0] if tir.ndim == 3 else tir

            # ── Mask ──────────────────────────────────────────────────────
            if not skip_masks:
                if overwrite or not os.path.exists(mask_path):
                    mask = make_mask(tir_2d)
                    np.save(mask_path, mask)
                    done_masks += 1

            # ── Edge ──────────────────────────────────────────────────────
            if not skip_edges:
                if overwrite or not os.path.exists(edge_path):
                    edge = make_edge(tir_2d)
                    np.save(edge_path, edge)
                    done_edges += 1

            if i % 500 == 0 or i == len(sample_dirs):
                print(f"  [{i:>5}/{len(sample_dirs)}] masks={done_masks} edges={done_edges} "
                      f"skipped={skipped} errors={errors}")

        except Exception as e:
            print(f"  [{i:>5}/{len(sample_dirs)}] ERROR in {sample_dir}: {e}")
            errors += 1

    print()
    print("=" * 60)
    print(f"  Done!")
    print(f"  Masks written  : {done_masks}")
    print(f"  Edges written  : {done_edges}")
    print(f"  Skipped        : {skipped}")
    print(f"  Errors         : {errors}")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Enrich existing patch output with semantic masks and Canny edges.'
    )
    parser.add_argument('--patches_dir', type=str, default='output/patches',
                        help='Root patches directory (default: output/patches)')
    parser.add_argument('--skip_masks',  action='store_true',
                        help='Skip mask generation (generate edges only)')
    parser.add_argument('--skip_edges',  action='store_true',
                        help='Skip edge generation (generate masks only)')
    parser.add_argument('--overwrite',   action='store_true',
                        help='Overwrite existing mask/edge files (default: skip if exists)')
    args = parser.parse_args()

    enrich_patches(
        patches_dir=args.patches_dir,
        skip_masks=args.skip_masks,
        skip_edges=args.skip_edges,
        overwrite=args.overwrite,
    )
