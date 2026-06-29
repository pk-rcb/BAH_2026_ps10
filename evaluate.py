"""
evaluate.py — Full pipeline evaluation on the held-out 10% test set.

Supports three colorization backends:
  --model pix2pix   GlobalGenerator      output in [-1, 1]
  --model spade     SPADEGenerator       output in [-1, 1]

Runs three stages:
  1. Rebuild the same test split (seed=42) used during training.
  2. SwinIR SR (optional via --two_stage) + colorization for every test sample.
  3. Compute PSNR / SSIM on-the-fly; save real + fake RGB PNGs for FID:
       python -m pytorch_fid eval_output/real_rgb eval_output/fake_rgb

IMPORTANT — output-range contract:
  - Pix2Pix / SPADE generators output [-1, 1]  (Tanh)
  - All metrics use [0, 1] after (x+1)/2 denormalisation
  - Saved PNGs are [0, 1] clipped — correct for FID
  - The hackathon .tif output (BGR swap) is done only in inference.py, NOT here

Usage:
    python evaluate.py --model spade \\
        --color_weights weights/best_spade_color_model.pth \\
        --patches_dir   output/patches
"""

import os
import argparse
import json
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image

from dataset import TIRDataset, N_MASK_CLASSES
from models import SwinIR, SPADEGenerator
from metrics import calculate_psnr, calculate_ssim_metric, calculate_rmse
import cv2

try:
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, DDIMScheduler
except ImportError:
    pass  # diffusers only needed if model == 'controlnet'


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_class, weights_path, device, **kwargs):
    model = model_class(**kwargs).to(device)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"  ✔ Loaded: {weights_path}")
    else:
        print(f"  ✘ WARNING: weights not found at {weights_path}. Using random init.")
    model.eval()
    return model


def tensor_to_01(t):
    """Map a tensor from [-1, 1] → [0, 1], clipped.  Safe for Tanh outputs."""
    return ((t + 1.0) / 2.0).clamp(0.0, 1.0)

def percentile_stretch_u8(arr: np.ndarray, low: int = 2, high: int = 98) -> np.ndarray:
    lo  = np.percentile(arr, low)
    hi  = np.percentile(arr, high)
    clipped = np.clip(arr, lo, hi)
    if hi - lo < 1e-5:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((clipped - lo) / (hi - lo) * 255.0).astype(np.uint8)

def get_edge_map(tir_img: np.ndarray) -> np.ndarray:
    """Takes a (H, W) float TIR array, stretches it, runs Canny, returns (H, W, 3) edges in [0, 1]."""
    tir_u8 = percentile_stretch_u8(tir_img)
    tir_blur = cv2.GaussianBlur(tir_u8, (3, 3), 0)
    edges = cv2.Canny(tir_blur, threshold1=50, threshold2=150)
    edges = (edges / 255.0).clip(0.0, 1.0)
    return np.stack([edges, edges, edges], axis=-1)

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(args):
    use_controlnet = (args.model == 'controlnet')
    use_spade = (args.model == 'spade')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Evaluation device : {device}")
    print(f"  Color model       : {args.model.upper()}")
    print(f"{'='*60}\n")

    # ── 1. Rebuild the SAME test split used during training ───────────────────
    color_stats_file = os.path.join(args.patches_dir, 'global_stats_color.json')

    full_dataset = TIRDataset(
        patches_dir=args.patches_dir,
        task='color',
        stats_file=color_stats_file
    )

    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size   = int(0.1 * total_size)
    test_size  = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(42)   # SAME seed as training
    _, _, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    print(f"Test set size: {test_size} samples ({len(test_loader)} batches)\n")

    # ── 2. Load models ─────────────────────────────────────────────────────────
    print("Loading models...")

    sr_model = load_model(
        SwinIR, args.sr_weights, device,
        in_channels=1, out_channels=1,
        embed_dim=96, depths=6, num_heads=6,
        window_size=8, mlp_ratio=4.0, upscale=2
    )

    if use_controlnet:
        print(f"Loading ControlNet from {args.controlnet_dir}...")
        controlnet = ControlNetModel.from_pretrained(args.controlnet_dir, torch_dtype=torch.float16)
        print(f"Loading SD backbone {args.model_id}...")
        color_model = StableDiffusionControlNetPipeline.from_pretrained(
            args.model_id, controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None
        )
        color_model.scheduler = DDIMScheduler.from_config(color_model.scheduler.config)
        color_model.to(device)
        color_model.set_progress_bar_config(disable=True)
    elif use_spade:
        color_model = load_model(
            SPADEGenerator, args.color_weights, device,
            tir_channels=1, label_nc=N_MASK_CLASSES,
            out_channels=3, ngf=64
        )
    else:
        raise ValueError("Invalid model selected")

    # ── 3. Output directories for FID ─────────────────────────────────────────
    real_dir = os.path.join(args.output_dir, "real_rgb")
    fake_dir = os.path.join(args.output_dir, "fake_rgb")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)
    print(f"\nSaving images to:\n  Real → {real_dir}\n  Fake → {fake_dir}\n")

    # ── 4. Evaluation loop ─────────────────────────────────────────────────────
    all_psnr   = []
    all_ssim   = []
    all_rmse   = []
    sample_idx = 0

    print(f"{'─'*60}")
    print(f"  Running inference on test set…")
    print(f"{'─'*60}")

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Dataset task='color' returns (tir, mask_onehot, rgb) — 3 items
            tir_input, mask_input, real_rgb = batch
            tir_input = tir_input.to(device)   # (B, 1, H, W)   [-1, 1]
            mask_input = mask_input.to(device) # (B, K, H, W)   one-hot float
            real_rgb   = real_rgb.to(device)   # (B, 3, H, W)   [-1, 1]

            # ── Optional Stage 1: SwinIR Super-Resolution ──────────────────
            if args.two_stage:
                tir_lr = F.interpolate(tir_input, scale_factor=0.5, mode='bicubic',
                                       align_corners=False, antialias=True)
                tir_input = sr_model(tir_lr)

            # ── Stage 2: Colorization ──────────────────────────────────────
            if use_controlnet:
                # 1. Generate edge maps for the batch
                tir_np = tir_input.cpu().numpy() # (B, 1, H, W)
                edge_list = []
                for b in range(tir_np.shape[0]):
                    edge_img = get_edge_map(tir_np[b, 0])
                    edge_list.append(edge_img)
                
                # 2. Run diffusion
                result = color_model(
                    prompt=[""] * len(edge_list),
                    image=edge_list,
                    num_inference_steps=args.inference_steps,
                    guidance_scale=1.0,
                    output_type="pt" # returns (B, 3, H, W) float in [0, 1]
                )
                fake_rgb = result.images.to(device)
            elif use_spade:
                fake_rgb = color_model(tir_input, mask_input)

            # ── Metrics ─────────────────────────────────────────────────────
            for i in range(fake_rgb.size(0)):
                # If ControlNet, fake_rgb is already in [0,1]. Otherwise map [-1,1] -> [0,1].
                if use_controlnet:
                    fake_i = fake_rgb[i].unsqueeze(0)
                else:
                    fake_i = tensor_to_01(fake_rgb[i].unsqueeze(0))
                real_i = tensor_to_01(real_rgb[i].unsqueeze(0))

                all_psnr.append(calculate_psnr(fake_i.clone(), real_i.clone()))
                all_ssim.append(calculate_ssim_metric(fake_i.clone(), real_i.clone()))
                all_rmse.append(calculate_rmse(fake_i.clone(), real_i.clone()))

                save_image(real_i, os.path.join(real_dir, f"sample_{sample_idx:05d}.png"))
                save_image(fake_i, os.path.join(fake_dir, f"sample_{sample_idx:05d}.png"))
                sample_idx += 1

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(test_loader):
                running_psnr = sum(all_psnr) / len(all_psnr)
                running_ssim = sum(all_ssim) / len(all_ssim)
                running_rmse = sum(all_rmse) / len(all_rmse)
                print(f"  [{batch_idx+1:>4d}/{len(test_loader)}]  "
                      f"PSNR: {running_psnr:.2f}  |  SSIM: {running_ssim:.4f}  |  RMSE: {running_rmse:.4f}")

    # ── 5. Final report ────────────────────────────────────────────────────────
    mean_psnr = sum(all_psnr) / len(all_psnr)
    mean_ssim = sum(all_ssim) / len(all_ssim)
    mean_rmse = sum(all_rmse) / len(all_rmse)

    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS  ({sample_idx} test samples)  [{args.model.upper()}]")
    print(f"{'='*60}")
    print(f"  Mean PSNR : {mean_psnr:.4f} dB")
    print(f"  Mean SSIM : {mean_ssim:.4f}")
    print(f"  Mean RMSE : {mean_rmse:.4f}")
    print(f"{'='*60}")
    print(f"\n  FID Calculation (run this next):")
    print(f"    pip install pytorch-fid")
    print(f"    python -m pytorch_fid {real_dir} {fake_dir}")
    print()

    results = {
        "model":           args.model,
        "num_samples":     sample_idx,
        "mean_psnr_db":    round(mean_psnr, 4),
        "mean_ssim":       round(mean_ssim, 4),
        "mean_rmse":       round(mean_rmse, 4),
        "two_stage_eval":  args.two_stage,
        "per_sample_psnr": [round(v, 4) for v in all_psnr],
        "per_sample_ssim": [round(v, 4) for v in all_ssim],
        "per_sample_rmse": [round(v, 4) for v in all_rmse],
    }
    results_path = os.path.join(args.output_dir, f"eval_results_{args.model}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved → {results_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the full SwinIR + colorization pipeline on the held-out test set."
    )
    parser.add_argument("--model",          type=str, default="spade",
                        choices=["spade", "controlnet"],
                        help="Colorization model architecture (default: spade)")
    parser.add_argument("--patches_dir",    type=str, default="output/patches")
    parser.add_argument("--sr_weights",     type=str, default="weights/best_sr_model.pth")
    parser.add_argument("--color_weights",  type=str, default="weights/best_spade_color_model.pth",
                        help="Path to SPADE weights")
    parser.add_argument("--controlnet_dir", type=str, default="weights/controlnet_color",
                        help="Path to saved ControlNet adapter")
    parser.add_argument("--model_id",       type=str, default="runwayml/stable-diffusion-v1-5",
                        help="HuggingFace model ID for diffusion backbone")
    parser.add_argument("--inference_steps",type=int, default=8,
                        help="Number of diffusion steps for ControlNet inference")
    parser.add_argument("--output_dir",     type=str, default="eval_output")
    parser.add_argument("--batch_size",     type=int, default=4)
    parser.add_argument("--num_workers",    type=int, default=2)
    parser.add_argument("--two_stage",      action="store_true",
                        help="Simulate full two-stage: bicubic-down TIR → SwinIR → colorize")

    args = parser.parse_args()
    evaluate(args)
