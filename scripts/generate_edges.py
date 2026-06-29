"""
generate_edges.py — Canny Edge Extractor for ControlNet Conditioning
=====================================================================
Reads a tir_100m.npy patch, applies percentile stretch → uint8,
then runs Canny edge detection to produce edge_100m.npy.

The edges act as structural constraints for the ControlNet diffusion model,
preventing hallucination of roads/coastlines in wrong locations.

Output: edge_100m.npy  — shape (H,W), dtype uint8, values {0, 255}
"""

import numpy as np
import os
import argparse
import logging
import cv2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def percentile_stretch_u8(arr: np.ndarray, low: int = 2, high: int = 98) -> np.ndarray:
    """Stretch a 2-D float array to uint8 [0, 255] using percentile clipping."""
    lo  = np.percentile(arr, low)
    hi  = np.percentile(arr, high)
    clipped = np.clip(arr, lo, hi)
    if hi - lo < 1e-5:
        return np.zeros_like(arr, dtype=np.uint8)
    stretched = (clipped - lo) / (hi - lo) * 255.0
    return stretched.astype(np.uint8)


def generate_edges(tir_path: str, output_path: str,
                   threshold1: int = 50, threshold2: int = 150) -> np.ndarray:
    """
    Args:
        tir_path    : Path to tir_100m.npy  —  shape (1,H,W) or (H,W), float32
        output_path : Where to save edge_100m.npy  —  shape (H,W), uint8 {0,255}
        threshold1  : Canny lower hysteresis threshold
        threshold2  : Canny upper hysteresis threshold

    Returns:
        edges  np.ndarray  shape (H,W), uint8
    """
    tir = np.load(tir_path).astype(np.float32)
    tir_2d = tir[0] if tir.ndim == 3 else tir

    # Stretch to uint8 for Canny (requires 8-bit input)
    tir_u8 = percentile_stretch_u8(tir_2d)

    # Optional: slight Gaussian blur to reduce noise speckle before edge detection
    tir_blur = cv2.GaussianBlur(tir_u8, (3, 3), 0)

    edges = cv2.Canny(tir_blur, threshold1=threshold1, threshold2=threshold2)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.save(output_path, edges)

    edge_density = 100.0 * (edges > 0).sum() / edges.size
    logger.info(f"Saved Canny edges (density={edge_density:.1f}%) → {output_path}")
    return edges


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract Canny edges from TIR data for ControlNet.')
    parser.add_argument('tir_path',    type=str, help='Path to tir_100m.npy')
    parser.add_argument('output_path', type=str, help='Path to save edge_100m.npy')
    parser.add_argument('--threshold1', type=int, default=50,  help='Canny lower threshold')
    parser.add_argument('--threshold2', type=int, default=150, help='Canny upper threshold')
    args = parser.parse_args()

    generate_edges(args.tir_path, args.output_path, args.threshold1, args.threshold2)
