import os
import argparse
import glob
import torch
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import cv2

from models import SwinIR, SPADEGenerator
from inference import load_model, process_image
from dataset import N_MASK_CLASSES

try:
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, DDIMScheduler
except ImportError:
    pass

def stretch(img, min_pct=2, max_pct=98):
    p_min, p_max = np.percentile(img, (min_pct, max_pct))
    if p_max - p_min < 1e-5: return np.zeros_like(img)
    stretched = (img - p_min) / (p_max - p_min)
    return np.clip(stretched, 0, 1)

def get_edge_map_u8(tir_img: np.ndarray) -> np.ndarray:
    """Diagnostic edge extraction."""
    tir_u8 = (stretch(tir_img) * 255.0).astype(np.uint8)
    tir_blur = cv2.GaussianBlur(tir_u8, (3, 3), 0)
    edges = cv2.Canny(tir_blur, threshold1=50, threshold2=150)
    return edges # (H, W) uint8

def visualize_test(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Diagnostics running on device: {device}")
    
    # ── 1. Load Models ──
    print("Loading SwinIR...")
    sr_model = load_model(
        SwinIR, args.sr_weights, device,
        in_channels=1, out_channels=1,
        embed_dim=96, depths=6, num_heads=6,
        window_size=8, mlp_ratio=4.0, upscale=2
    )
    
    print("Loading SPADE...")
    spade_model = load_model(
        SPADEGenerator, args.spade_weights, device,
        tir_channels=1, label_nc=N_MASK_CLASSES, out_channels=3, ngf=64
    )
    
    print("Loading ControlNet...")
    controlnet = ControlNetModel.from_pretrained(args.controlnet_dir, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.model_id, controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    
    # ── 2. Get Test Image ──
    search_path = os.path.join(args.input_dir, '*_tir_200m.tif')
    input_files = glob.glob(search_path)
    if not input_files: return print(f"No test images found in {search_path}")
    test_file = input_files[0]
    print(f"Testing on: {os.path.basename(test_file)}")
    
    tir_200m = tifffile.imread(test_file).astype(np.float32)
    
    # ── 3. Super-Resolution ──
    print("Running SwinIR...")
    sr_output = process_image(sr_model, tir_200m, patch_size=128, upsample_factor=2, device=device)
    sr_img = sr_output[0] # (H, W)
    
    # ── 4. SPADE Colorization ──
    print("Running SPADE...")
    spade_output = process_image(spade_model, sr_img, patch_size=256, upsample_factor=1, device=device, use_spade=True)
    spade_rgb = np.transpose(spade_output, (1, 2, 0)) # (H, W, 3) in [0, 1]
    
    # ── 5. ControlNet Colorization ──
    print("Running ControlNet...")
    edges_u8 = get_edge_map_u8(sr_img)
    edges_rgb = np.stack([edges_u8, edges_u8, edges_u8], axis=-1) # (H, W, 3)
    
    import PIL.Image
    edges_pil = PIL.Image.fromarray(edges_rgb)
    
    result = pipe(
        prompt="", 
        image=edges_pil,
        num_inference_steps=20,
        guidance_scale=1.0,
        output_type="np"
    )
    cnet_rgb = result.images[0] # (H, W, 3) in [0, 1]
    
    # ── 6. Plotting ──
    fig, axes = plt.subplots(1, 5, figsize=(24, 6))
    
    axes[0].imshow(stretch(tir_200m), cmap='gray')
    axes[0].set_title("1. Input TIR (200m)")
    axes[0].axis('off')

    axes[1].imshow(stretch(sr_img), cmap='gray')
    axes[1].set_title("2. SwinIR (100m)")
    axes[1].axis('off')
    
    axes[2].imshow(edges_rgb)
    axes[2].set_title("3. Canny Edge Map (ControlNet Input)")
    axes[2].axis('off')
    
    axes[3].imshow(np.clip(spade_rgb, 0, 1))
    axes[3].set_title("4. SPADE RGB")
    axes[3].axis('off')
    
    axes[4].imshow(np.clip(cnet_rgb, 0, 1))
    axes[4].set_title("5. ControlNet RGB")
    axes[4].axis('off')
    
    plt.tight_layout()
    save_path = 'diagnostic_results.png'
    plt.savefig(save_path, dpi=300)
    print(f"Diagnostic visualization saved to {os.path.abspath(save_path)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='output/downscaled_data')
    parser.add_argument('--sr_weights', type=str, default='weights/best_sr_model.pth')
    parser.add_argument('--spade_weights', type=str, default='weights/best_spade_color_model.pth')
    parser.add_argument('--controlnet_dir', type=str, default='weights/controlnet_color')
    parser.add_argument('--model_id', type=str, default='runwayml/stable-diffusion-v1-5')
    args, unknown = parser.parse_known_args()
    visualize_test(args)
