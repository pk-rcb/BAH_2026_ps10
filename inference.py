"""
inference.py — Full pipeline inference: SwinIR SR → colorization → .tif output
================================================================================
Supports one colorization backend:
    --model spade     SPADEGenerator    (semantic-mask conditioned)

OUTPUT CONTRACT:
  - Models output RGB in [-1, 1]  (Tanh).
  - This script denormalises to [0, 1], then saves as the hackathon's required
    BGR channel order  (Layer1=Blue, Layer2=Green, Layer3=Red).
  - The BGR swap happens ONLY here, at the very last step before tifffile.imwrite.
  - Training data and evaluation metrics always use RGB.

Usage:
    # SPADE:
    python inference.py --model spade \\
        --color_weights weights/best_spade_color_model.pth
"""

import os
import argparse
import glob
import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

from models import SwinIR, SPADEGenerator
from dataset import N_MASK_CLASSES
from utils.file_utils import find_file

N_CLUSTERS = N_MASK_CLASSES   # 4: water / vegetation / urban / bare-rock


# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_class, weights_path, device, **kwargs):
    model = model_class(**kwargs).to(device)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"Warning: Weights not found at {weights_path}. Using uninitialized model.")
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# On-the-fly mask generation (for SPADE inference — no pre-saved mask needed)
# ─────────────────────────────────────────────────────────────────────────────

def _make_mask_onehot(tir_patch: np.ndarray, n_clusters: int = N_CLUSTERS,
                      device: torch.device = None) -> torch.Tensor:
    """
    Runs K-Means on a single TIR patch and returns a one-hot mask tensor.

    Args:
        tir_patch : np.ndarray  shape (1, H, W) or (H, W),  float32 in [-1,1]
        n_clusters : int  number of K-Means classes (default 4)
        device    : torch device

    Returns:
        torch.Tensor  shape (1, K, H, W) float32  — one-hot, ready for SPADE
    """
    arr = tir_patch[0] if tir_patch.ndim == 3 else tir_patch
    H, W = arr.shape
    flat = arr.reshape(-1, 1)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    raw = km.fit_predict(flat)

    # Sort by centroid temperature (label 0 = coldest = water)
    order = np.argsort(km.cluster_centers_.flatten())
    remap = np.empty(n_clusters, dtype=np.int32)
    for new_lbl, old_lbl in enumerate(order):
        remap[old_lbl] = new_lbl
    mask = remap[raw].reshape(H, W).astype(np.int64)

    mask_t  = torch.from_numpy(mask)                            # (H, W)
    onehot  = F.one_hot(mask_t, num_classes=n_clusters)        # (H, W, K)
    onehot  = onehot.permute(2, 0, 1).float().unsqueeze(0)     # (1, K, H, W)
    return onehot.to(device) if device else onehot


# ─────────────────────────────────────────────────────────────────────────────
# Patch-by-patch processing
# ─────────────────────────────────────────────────────────────────────────────

def process_image(color_model, sr_model, img, patch_size, upsample_factor,
                  device, use_spade=False):
    """
    Run SR + colorization on a full image patch-by-patch.

    Args:
        color_model    : GlobalGenerator or SPADEGenerator
        sr_model       : SwinIR (set to None to skip SR)
        img            : np.ndarray  (H,W) or (1,H,W), float32 raw TIR
        patch_size     : spatial size of each input patch
        upsample_factor: 2 for SR, 1 for colorization
        use_spade      : bool  — whether to pass mask to generator

    Returns:
        np.ndarray  (C_out, H_out, W_out) in [0, 1]  — RGB order
    """
    if img.ndim == 2:
        img = np.expand_dims(img, 0)

    _, orig_h, orig_w = img.shape
    out_h = orig_h * upsample_factor
    out_w = orig_w * upsample_factor
    out_c = 3

    # Pad so spatial dims are multiples of patch_size
    pad_h = (patch_size - orig_h % patch_size) % patch_size
    pad_w = (patch_size - orig_w % patch_size) % patch_size
    img_pad = np.pad(img, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
    _, ph, pw = img_pad.shape

    out_hp = ph * upsample_factor
    out_wp = pw * upsample_factor
    output = np.zeros((out_c, out_hp, out_wp), dtype=np.float32)

    # Normalise input to [-1, 1]
    i_min, i_max = img_pad.min(), img_pad.max()
    img_norm = 2.0 * ((img_pad - i_min) / max(i_max - i_min, 1e-5)) - 1.0

    stride = patch_size
    with torch.no_grad():
        for y in range(0, ph, stride):
            for x in range(0, pw, stride):
                patch     = img_norm[:, y:y + patch_size, x:x + patch_size]
                patch_t   = torch.from_numpy(patch).unsqueeze(0).float().to(device)

                if sr_model is not None:
                    patch_t = sr_model(patch_t)   # (1,1,H*2,W*2) in [-1,1]

                if use_spade:
                    mask_t = _make_mask_onehot(patch_t.squeeze(0).cpu().numpy(),
                                               n_clusters=N_CLUSTERS, device=device)
                    out_p = color_model(patch_t, mask_t)  # (1,3,H,W) in [-1,1]
                else:
                    out_p = color_model(patch_t)           # (1,3,H,W) in [-1,1]

                # Denormalise [-1,1] → [0,1]
                out_np = ((out_p.squeeze(0).cpu().numpy() + 1.0) / 2.0).clip(0.0, 1.0)

                oy = y * upsample_factor
                ox = x * upsample_factor
                bs = patch_size * upsample_factor
                output[:, oy:oy + bs, ox:ox + bs] = out_np

    return output[:, :out_h, :out_w]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_spade = True
    print(f"Inference using device: {device} | model: {args.model.upper()}")

    # ── Load SwinIR ─────────────────────────────────────────────────────────
    sr_model = load_model(
        SwinIR, args.sr_weights, device,
        in_channels=1, out_channels=1,
        embed_dim=96, depths=6, num_heads=6,
        window_size=8, mlp_ratio=4.0, upscale=2
    )

    # ── Load colorization model ──────────────────────────────────────────────
    color_model = load_model(
        SPADEGenerator, args.color_weights, device,
        tir_channels=1, label_nc=N_MASK_CLASSES, out_channels=3, ngf=64
    )

    # ── Output directories ───────────────────────────────────────────────────
    out_sr_dir    = os.path.join(args.output_dir, 'model_outputs', 'tir_superresolved_100m')
    out_color_dir = os.path.join(args.output_dir, 'model_outputs', 'colorized_tir_100m')
    os.makedirs(out_sr_dir,    exist_ok=True)
    os.makedirs(out_color_dir, exist_ok=True)

    # ── Find input files ─────────────────────────────────────────────────────
    input_files = glob.glob(os.path.join(args.input_dir, '*_tir_200m.tif'))
    if not input_files:
        print(f"No files found matching {args.input_dir}/*_tir_200m.tif")
        return

    for file_path in input_files:
        filename   = os.path.basename(file_path)
        product_id = filename.split('_')[0]
        print(f"\nProcessing product: {product_id}")

        tir_200m = tifffile.imread(file_path).astype(np.float32)

        # ── Stage 1: SwinIR Super-Resolution 200m → 100m ────────────────────
        print("  [1/2] SwinIR Super-Resolution...")
        sr_out = process_image(
            color_model=None, sr_model=sr_model,
            img=tir_200m, patch_size=128, upsample_factor=2,
            device=device, use_spade=False
        )
        # sr_out: (1, H, W) in [0, 1]
        sr_save = (sr_out[0] * 65535.0).clip(0, 65535).astype(np.uint16)
        tifffile.imwrite(os.path.join(out_sr_dir, f'{product_id}.tif'), sr_save)
        print(f"     Saved → {out_sr_dir}/{product_id}.tif")

        # ── Stage 2: Colorization 100m TIR → RGB ────────────────────────────
        print(f"  [2/2] {args.model.upper()} Colorization...")
        color_out = process_image(
            color_model=color_model, sr_model=None,
            img=sr_out[0], patch_size=256, upsample_factor=1,
            device=device, use_spade=use_spade
        )
        # color_out: (3, H, W) in [0, 1], RGB order from model

        # ── BGR SWAP — required by hackathon spec ────────────────────────────
        # Model trains in RGB. Spec requires: Layer1=Blue, Layer2=Green, Layer3=Red.
        # Swap here, at the LAST step before saving. Never do this in training.
        r, g, b   = color_out[0], color_out[1], color_out[2]
        bgr_output = np.stack([b, g, r], axis=0)      # [Blue, Green, Red]
        bgr_u8     = (bgr_output * 255.0).clip(0, 255).astype(np.uint8)
        tifffile.imwrite(
            os.path.join(out_color_dir, f'{product_id}.tif'),
            bgr_u8, photometric='rgb'
        )
        print(f"     Saved → {out_color_dir}/{product_id}.tif  (BGR as per spec)")

    print("\nInference complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SwinIR + Colorization inference pipeline')
    parser.add_argument('--model',         type=str, default='spade',
                        choices=['spade'],
                        help='Colorization model architecture (default: spade)')
    parser.add_argument('--input_dir',     type=str, default='output/downscaled_data',
                        help='Directory containing *_tir_200m.tif input files')
    parser.add_argument('--output_dir',    type=str, default='output')
    parser.add_argument('--sr_weights',    type=str, default='weights/best_sr_model.pth')
    parser.add_argument('--color_weights', type=str, default='weights/best_spade_color_model.pth',
                        help='Colorization weights path '
                             '(e.g. weights/best_spade_color_model.pth for SPADE)')
    args = parser.parse_args()
    main(args)
