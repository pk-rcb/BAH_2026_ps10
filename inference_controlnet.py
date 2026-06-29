"""
inference_controlnet.py — ControlNet Diffusion Inference Pipeline
===================================================================
Runs SwinIR SR (optional) → Canny Edge Extraction → ControlNet Diffusion.
Uses the DDIM scheduler for fast inference (4-8 steps).

OUTPUT CONTRACT:
  - Diffusion outputs RGB in [0, 1].
  - Saved to .tif as BGR (Layer1=Blue, Layer2=Green, Layer3=Red) as per hackathon spec.

Usage:
    python inference_controlnet.py \
        --model_id runwayml/stable-diffusion-v1-5 \
        --controlnet_dir weights/controlnet \
        --input_dir output/downscaled_data \
        --num_inference_steps 8
"""

import os
import argparse
import glob
import numpy as np
import tifffile
import torch
from PIL import Image

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler
from models import SwinIR
from inference import load_model, process_image as process_image_base
from scripts.generate_edges import generate_edges, percentile_stretch_u8
import cv2


# ─────────────────────────────────────────────────────────────────────────────
# Edge Extraction
# ─────────────────────────────────────────────────────────────────────────────
def get_edge_map(tir_patch: np.ndarray, threshold1=50, threshold2=150) -> Image.Image:
    """
    tir_patch: (1, H, W) or (H, W) in [0, 1] (or [-1,1], but we stretch it anyway).
    Returns PIL Image of Canny edges in RGB format (as expected by diffusers).
    """
    arr = tir_patch[0] if tir_patch.ndim == 3 else tir_patch
    u8 = percentile_stretch_u8(arr)
    blur = cv2.GaussianBlur(u8, (3, 3), 0)
    edges = cv2.Canny(blur, threshold1, threshold2)
    
    # diffusers pipeline expects RGB PIL Image for controlnet_cond
    edges_rgb = np.stack([edges, edges, edges], axis=-1)
    return Image.fromarray(edges_rgb)


# ─────────────────────────────────────────────────────────────────────────────
# Patch Processing for Diffusion
# ─────────────────────────────────────────────────────────────────────────────
def process_image_diffusion(pipeline, sr_model, img, patch_size, upsample_factor, 
                            device, num_inference_steps=8):
    """
    Similar to inference.py's process_image, but adapted for the diffusers pipeline.
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

    # Normalise input to [-1, 1] for SR model compatibility
    i_min, i_max = img_pad.min(), img_pad.max()
    img_norm = 2.0 * ((img_pad - i_min) / max(i_max - i_min, 1e-5)) - 1.0

    stride = patch_size
    
    for y in range(0, ph, stride):
        for x in range(0, pw, stride):
            patch = img_norm[:, y:y + patch_size, x:x + patch_size]
            patch_t = torch.from_numpy(patch).unsqueeze(0).float().to(device)

            with torch.no_grad():
                if sr_model is not None:
                    patch_t = sr_model(patch_t)  # (1, 1, H*2, W*2) in [-1, 1]

            # Get edge map
            # patch_t is in [-1, 1], get_edge_map applies percentile stretch anyway
            edge_img = get_edge_map(patch_t.squeeze(0).cpu().numpy())

            # Diffusion inference
            # We use an empty prompt.
            result = pipeline(
                prompt="",
                image=edge_img,
                num_inference_steps=num_inference_steps,
                guidance_scale=1.0,  # no classifier-free guidance for speed/empty prompt
                output_type="np"     # returns list of numpy arrays in [0, 1], shape (H, W, 3)
            )
            
            # (H, W, 3) -> (3, H, W)
            out_np = result.images[0].transpose(2, 0, 1)

            oy = y * upsample_factor
            ox = x * upsample_factor
            bs = patch_size * upsample_factor
            output[:, oy:oy + bs, ox:ox + bs] = out_np

    return output[:, :out_h, :out_w]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Inference using device: {device}")

    # ── 1. Load SwinIR ────────────────────────────────────────────────────────
    sr_model = load_model(
        SwinIR, args.sr_weights, device,
        in_channels=1, out_channels=1,
        embed_dim=96, depths=6, num_heads=6,
        window_size=8, mlp_ratio=4.0, upscale=2
    )

    # ── 2. Load Diffusion Pipeline ───────────────────────────────────────────
    print(f"Loading ControlNet from {args.controlnet_dir}...")
    controlnet = ControlNetModel.from_pretrained(args.controlnet_dir, torch_dtype=torch.float16)
    
    print(f"Loading SD backbone {args.model_id}...")
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.model_id,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None
    )
    
    # Use DDIM for fast inference (4-8 steps)
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    
    # Enable memory efficient attention if possible
    # try:
    #     pipeline.enable_xformers_memory_efficient_attention()
    # except ImportError:
    #     pass

    # ── 3. Output directories ────────────────────────────────────────────────
    out_sr_dir    = os.path.join(args.output_dir, 'model_outputs', 'tir_superresolved_100m')
    out_color_dir = os.path.join(args.output_dir, 'model_outputs', 'colorized_tir_100m')
    os.makedirs(out_sr_dir,    exist_ok=True)
    os.makedirs(out_color_dir, exist_ok=True)

    # ── 4. Process files ─────────────────────────────────────────────────────
    input_files = glob.glob(os.path.join(args.input_dir, '*_tir_200m.tif'))
    if not input_files:
        print(f"No files found matching {args.input_dir}/*_tir_200m.tif")
        return

    for file_path in input_files:
        filename   = os.path.basename(file_path)
        product_id = filename.split('_')[0]
        print(f"\nProcessing product: {product_id}")

        tir_200m = tifffile.imread(file_path).astype(np.float32)

        print("  [1/2] SwinIR Super-Resolution...")
        sr_out = process_image_base(
            color_model=None, sr_model=sr_model,
            img=tir_200m, patch_size=128, upsample_factor=2,
            device=device, use_spade=False
        )
        sr_save = (sr_out[0] * 65535.0).clip(0, 65535).astype(np.uint16)
        tifffile.imwrite(os.path.join(out_sr_dir, f'{product_id}.tif'), sr_save)
        print(f"     Saved → {out_sr_dir}/{product_id}.tif")

        print(f"  [2/2] ControlNet Diffusion Colorization ({args.num_inference_steps} steps)...")
        # Diffusers typically works on 512x512 patches. SwinIR output patches will be 256x256.
        # We'll use 256x256 patches here (SD-1.5 can handle 256x256, though 512 is optimal).
        color_out = process_image_diffusion(
            pipeline=pipeline, sr_model=None,
            img=sr_out[0], patch_size=256, upsample_factor=1,
            device=device, num_inference_steps=args.num_inference_steps
        )

        # ── BGR SWAP ─────────────────────────────────────────────────────────
        # Output is in [0, 1], RGB format from the pipeline.
        r, g, b   = color_out[0], color_out[1], color_out[2]
        bgr_output = np.stack([b, g, r], axis=0)
        bgr_u8     = (bgr_output * 255.0).clip(0, 255).astype(np.uint8)
        tifffile.imwrite(
            os.path.join(out_color_dir, f'{product_id}.tif'),
            bgr_u8, photometric='rgb'
        )
        print(f"     Saved → {out_color_dir}/{product_id}.tif  (BGR as per spec)")

    print("\nInference complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ControlNet Diffusion inference pipeline')
    parser.add_argument('--model_id',      type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--input_dir',     type=str, default='output/downscaled_data')
    parser.add_argument('--output_dir',    type=str, default='output')
    parser.add_argument('--sr_weights',    type=str, default='weights/best_sr_model.pth')
    parser.add_argument('--controlnet_dir',type=str, default='weights/controlnet')
    parser.add_argument('--num_inference_steps', type=int, default=8,
                        help='Number of DDIM steps (4-10 recommended for speed)')
    args = parser.parse_args()
    main(args)
