import os
import argparse
import glob
import torch
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from models import SwinIR, GlobalGenerator
from inference import load_model, process_image

def stretch(img, min_pct=2, max_pct=98):
    """Percentile stretch for better visualization of TIR bands."""
    p_min, p_max = np.percentile(img, (min_pct, max_pct))
    stretched = (img - p_min) / (p_max - p_min)
    return np.clip(stretched, 0, 1)

def visualize_test(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    
    # Check if weights exist locally
    if not os.path.exists(args.sr_weights) or not os.path.exists(args.color_weights):
        print("ERROR: Model weights not found!")
        print(f"Please ensure you have downloaded your trained weights from Colab")
        print(f"and placed them in the 'weights/' folder.")
        print(f"Expected: {args.sr_weights} and {args.color_weights}")
        return

    sr_model = load_model(
        SwinIR, args.sr_weights, device,
        in_channels=1, out_channels=1,
        embed_dim=96, depths=6, num_heads=6,
        window_size=8, mlp_ratio=4.0, upscale=2
    )
    color_model = load_model(
        GlobalGenerator, args.color_weights, device,
        in_channels=1, out_channels=3, ngf=64, n_blocks=9
    )
    
    # Get a test image
    search_path = os.path.join(args.input_dir, '*_tir_200m.tif')
    input_files = glob.glob(search_path)
    
    if not input_files:
        print(f"No test images found in {search_path}")
        return
        
    test_file = input_files[0] # Just test on the first one
    print(f"Testing models on: {os.path.basename(test_file)}")
    
    # 1. Load Raw 200m TIR
    tir_200m = tifffile.imread(test_file).astype(np.float32)
    
    # 2. Run Super-Resolution
    print("Running Super-Resolution...")
    sr_output = process_image(sr_model, tir_200m, patch_size=128, upsample_factor=2, device=device)
    sr_img = sr_output[0]
    
    # 3. Run Colorization
    print("Running Colorization...")
    color_output = process_image(color_model, sr_img, patch_size=256, upsample_factor=1, device=device)
    
    # Colorization output is trained on [Red, Green, Blue] order
    # Matplotlib expects [Red, Green, Blue] as well (H, W, C)
    rgb_img = np.transpose(color_output, (1, 2, 0)) 
    
    # Prepare for visualization
    tir_200m_viz = stretch(tir_200m)
    sr_img_viz = stretch(sr_img)
    rgb_img_viz = np.clip(rgb_img, 0, 1) # Neural net outputs [0, 1] after denormalization
    
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(tir_200m_viz.squeeze(), cmap='gray')
    axes[0].set_title("Input: Raw TIR (200m)")
    axes[0].axis('off')

    axes[1].imshow(sr_img_viz.squeeze(), cmap='gray')
    axes[1].set_title("Stage 1: SwinIR Super-Resolved TIR (100m)")
    axes[1].axis('off')
    
    axes[2].imshow(rgb_img_viz)
    axes[2].set_title("Stage 2: Pix2PixHD Colorized RGB (100m)")
    axes[2].axis('off')
    
    plt.tight_layout()
    save_path = 'sample_results.png'
    plt.savefig(save_path, dpi=300)
    print(f"Success! Visualization saved to {os.path.abspath(save_path)}")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='output/downscaled_data', help='Directory containing 200m TIR images')
    parser.add_argument('--sr_weights', type=str, default='weights/best_sr_model.pth', help='Path to SR weights')
    parser.add_argument('--color_weights', type=str, default='weights/best_color_model.pth', help='Path to Color weights')
    args = parser.parse_args()
    
    visualize_test(args)
