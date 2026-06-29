import torch
import math
import numpy as np

def calculate_psnr(img1, img2):
    """
    Calculates PSNR (Peak Signal-to-Noise Ratio).
    Expects input tensors in the range [-1, 1] or [0, 1].
    """
    # Normalize to [0, 1] if they are in [-1, 1]
    if img1.min() < 0 or img2.min() < 0:
        img1 = (img1 + 1.0) / 2.0
        img2 = (img2 + 1.0) / 2.0
        
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    
    # Max pixel value is 1.0 after normalization
    max_pixel = 1.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

def calculate_ssim_metric(img1, img2):
    """
    Calculates SSIM.
    For inference metrics, it's recommended to use the same SSIM function
    we defined in models.py (which uses a gaussian window).
    """
    from models import SSIMLoss
    ssim_loss_fn = SSIMLoss(size_average=True).to(img1.device)
    
    # ssim_loss returns (1 - ssim), so ssim = 1 - loss
    loss = ssim_loss_fn(img1, img2)
    return 1.0 - loss.item()

def evaluate_fid(real_images_dir, generated_images_dir):
    """
    Evaluates Frechet Inception Distance (FID).
    Note: The most standard way to calculate FID is using the `pytorch-fid` package.
    Run: `pip install pytorch-fid`
    Then calculate via command line:
    `python -m pytorch_fid path/to/real path/to/generated`
    """
    print("To calculate FID, please install the pytorch-fid package:")
    print("pip install pytorch-fid")
    print(f"Then run: python -m pytorch_fid {real_images_dir} {generated_images_dir}")
