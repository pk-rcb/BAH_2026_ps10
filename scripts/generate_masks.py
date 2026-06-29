"""
generate_masks.py — K-Means Pseudo-Semantic Mask Generator
===========================================================
Reads a tir_100m.npy patch, runs K-Means (K=4) clustering on pixel
temperatures, and writes a mask_100m.npy with temperature-ordered labels:

    Label 0 (cold)  → water / rivers / lakes
    Label 1 (cool)  → dense vegetation / forests
    Label 2 (warm)  → urban areas / roads / buildings
    Label 3 (hot)   → bare soil / desert / exposed rock

The labels are always sorted by ascending centroid temperature so that
label 0 is guaranteed to be the coldest class regardless of K-Means init.
"""

import numpy as np
import os
import argparse
import logging
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

LABEL_NAMES = {
    0: "cold  (water/rivers)",
    1: "cool  (vegetation/forest)",
    2: "warm  (urban/roads)",
    3: "hot   (bare-soil/desert/rock)",
}


def generate_mask(tir_path: str, output_path: str, n_clusters: int = 4) -> np.ndarray:
    """
    Args:
        tir_path    : Path to tir_100m.npy  —  shape (1,H,W) or (H,W), float32
        output_path : Where to save mask_100m.npy  —  shape (H,W), uint8
        n_clusters  : Number of K-Means clusters (default 4)

    Returns:
        mask  np.ndarray  shape (H,W), values in {0 … n_clusters-1}
    """
    tir = np.load(tir_path).astype(np.float32)
    if tir.ndim == 3:
        tir_2d = tir[0]          # (1, H, W) → (H, W)
    elif tir.ndim == 2:
        tir_2d = tir
    else:
        raise ValueError(f"Unexpected TIR shape: {tir.shape}")

    H, W = tir_2d.shape
    flat = tir_2d.reshape(-1, 1)

    # K-Means on raw temperature values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    raw_labels = kmeans.fit_predict(flat)           # shape (H*W,)

    # Re-order labels so label 0 = coldest centroid, label K-1 = hottest
    centroids   = kmeans.cluster_centers_.flatten()
    sorted_idx  = np.argsort(centroids)             # ascending temperature
    remap       = np.empty(n_clusters, dtype=np.int32)
    for new_lbl, old_lbl in enumerate(sorted_idx):
        remap[old_lbl] = new_lbl

    mask = remap[raw_labels].reshape(H, W).astype(np.uint8)

    # Log cluster statistics
    for lbl in range(n_clusters):
        pct = 100.0 * (mask == lbl).sum() / mask.size
        centroid_temp = centroids[sorted_idx[lbl]]
        logger.debug(
            f"  Label {lbl} {LABEL_NAMES.get(lbl, '')} | "
            f"centroid={centroid_temp:.1f} | {pct:.1f}% of pixels"
        )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.save(output_path, mask)
    logger.info(f"Saved mask ({n_clusters} classes) → {output_path}")
    return mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate K-Means pseudo-semantic masks from TIR data.')
    parser.add_argument('tir_path',    type=str, help='Path to tir_100m.npy')
    parser.add_argument('output_path', type=str, help='Path to save mask_100m.npy')
    parser.add_argument('--n_clusters', type=int, default=4,
                        help='Number of K-Means clusters (default 4: water/veg/urban/rock)')
    args = parser.parse_args()

    generate_mask(args.tir_path, args.output_path, args.n_clusters)
