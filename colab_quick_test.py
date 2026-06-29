"""
colab_quick_test.py — Inline inference for Colab (SwinIR + SPADE + ControlNet)
================================================================================
Copy-paste the code inside the triple-quoted block below into a NEW CELL
in your Colab notebook (after your training cells).

It will:
  1. Prompt you to upload any TIR .npy file from your computer
  2. Run SwinIR Super-Resolution (TIR x2 upscale)
  3. Run SPADE Colorization (if weights exist)
  4. Run ControlNet Colorization (if weights exist)
  5. Display a beautiful comparison plot inline and download it.
"""

# ════════════════════════════════════════════════════════════════════
# PASTE EVERYTHING BELOW THIS LINE INTO A NEW COLAB CELL
# ════════════════════════════════════════════════════════════════════

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from google.colab import files
import cv2
from PIL import Image
from sklearn.cluster import KMeans
import tifffile

# ── Make sure we're in the project directory ──────────────────────
PROJECT_DIR = '/content/IR-colorization-BAH2026'
if os.getcwd() != PROJECT_DIR:
    os.chdir(PROJECT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from models import SwinIR, SPADEGenerator
from dataset import N_MASK_CLASSES

# ── Config ────────────────────────────────────────────────────────
SR_WEIGHTS         = 'weights/best_sr_model.pth'
SPADE_WEIGHTS      = 'weights/best_spade_color_model.pth'
CONTROLNET_DIR     = 'weights/controlnet'
SD_MODEL_ID        = 'runwayml/stable-diffusion-v1-5'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ─────────────────────────────────────────────────────────────────
# 1. Upload TIR .npy or .tif
# ─────────────────────────────────────────────────────────────────
print('\nUpload your TIR file (e.g. tir_200m.npy or tir_200m.tif):')
uploaded = files.upload()
if not uploaded:
    print("No file uploaded. Stopping.")
    sys.exit()
    
tif_path = list(uploaded.keys())[0]
print(f'Loaded: {tif_path}')

# ─────────────────────────────────────────────────────────────────
# 2. Helper Functions
# ─────────────────────────────────────────────────────────────────
def normalize_patch(arr):
    lo, hi = arr.min(), arr.max()
    if hi - lo > 1e-6:
        return 2.0 * ((arr - lo) / (hi - lo)) - 1.0
    return np.zeros_like(arr)

def _make_mask_onehot(tir_patch, n_clusters=4):
    arr = tir_patch[0] if tir_patch.ndim == 3 else tir_patch
    H, W = arr.shape
    flat = arr.reshape(-1, 1).astype(np.float32)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    raw = km.fit_predict(flat)
    order = np.argsort(km.cluster_centers_.flatten())
    remap = np.empty(n_clusters, dtype=np.int32)
    for new_lbl, old_lbl in enumerate(order):
        remap[old_lbl] = new_lbl
    mask = remap[raw].reshape(H, W).astype(np.int64)
    mask_t = torch.from_numpy(mask)
    onehot = F.one_hot(mask_t, num_classes=n_clusters).permute(2, 0, 1).float().unsqueeze(0)
    return onehot.to(device)

def get_edge_map(arr):
    arr = arr[0] if arr.ndim == 3 else arr
    lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
    if hi - lo < 1e-5:
        u8 = np.zeros_like(arr, dtype=np.uint8)
    else:
        u8 = np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    blur = cv2.GaussianBlur(u8, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges_rgb = np.stack([edges, edges, edges], axis=-1)
    return Image.fromarray(edges_rgb)

# ─────────────────────────────────────────────────────────────────
# 3. Load Models
# ─────────────────────────────────────────────────────────────────
print('\nLoading SwinIR...')
sr_model = SwinIR(in_channels=1, out_channels=1, embed_dim=96, depths=6, num_heads=6, window_size=8, mlp_ratio=4.0, upscale=2).to(device)
if os.path.exists(SR_WEIGHTS): sr_model.load_state_dict(torch.load(SR_WEIGHTS, map_location=device))
sr_model.eval()

has_spade = os.path.exists(SPADE_WEIGHTS)
if has_spade:
    print('Loading SPADEGenerator...')
    spade_model = SPADEGenerator(tir_channels=1, label_nc=N_MASK_CLASSES, out_channels=3, ngf=64).to(device)
    spade_model.load_state_dict(torch.load(SPADE_WEIGHTS, map_location=device))
    spade_model.eval()
else:
    print(f'SPADE weights not found at {SPADE_WEIGHTS}. Skipping SPADE.')

has_cnet = os.path.exists(CONTROLNET_DIR)
if has_cnet:
    print('Loading ControlNet & Stable Diffusion (this takes a moment)...')
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler
    cnet = ControlNetModel.from_pretrained(CONTROLNET_DIR, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(SD_MODEL_ID, controlnet=cnet, torch_dtype=torch.float16, safety_checker=None)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
else:
    print(f'ControlNet folder not found at {CONTROLNET_DIR}. Skipping ControlNet.')

# ─────────────────────────────────────────────────────────────────
# 4. Process
# ─────────────────────────────────────────────────────────────────
if tif_path.lower().endswith(('.tif', '.tiff')):
    tir_raw = tifffile.imread(tif_path).astype(np.float32)
else:
    tir_raw = np.load(tif_path).astype(np.float32)

if tir_raw.ndim == 3: tir_raw = tir_raw[0]
tir_norm = normalize_patch(tir_raw[np.newaxis, ...])

print('\nRunning SwinIR super-resolution...')
sr_img_padded = None
with torch.no_grad():
    t = torch.from_numpy(tir_norm).unsqueeze(0).float().to(device)
    
    # Pad to multiple of window_size (8) for SwinIR
    _, _, h, w = t.size()
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    if pad_h > 0 or pad_w > 0:
        t = F.pad(t, (0, pad_w, 0, pad_h), mode='reflect')
        
    sr_out_t = sr_model(t)
    
    # Crop back to original size (x2 for SR)
    if pad_h > 0 or pad_w > 0:
        sr_out_t = sr_out_t[:, :, :h*2, :w*2]
        
    sr_img = sr_out_t.squeeze(0).cpu().numpy()[0]
    sr_img = (sr_img + 1.0) / 2.0  # [0,1]

spade_rgb = None
if has_spade:
    print('Running SPADE colorization...')
    with torch.no_grad():
        sr_norm_t = (torch.from_numpy(sr_img).unsqueeze(0).unsqueeze(0).float().to(device) * 2.0) - 1.0
        mask_t = _make_mask_onehot(sr_norm_t.squeeze(0).cpu().numpy())
        spade_out_t = spade_model(sr_norm_t, mask_t)
        spade_rgb = ((spade_out_t.squeeze(0).cpu().numpy() + 1.0) / 2.0).clip(0, 1).transpose(1, 2, 0)

cnet_rgb = None
if has_cnet:
    print('Running ControlNet colorization...')
    # Diffusers works best with 512x512, we will resize SR to 512, process, and resize back to original SR size
    edge_map = get_edge_map(sr_img)
    orig_size = edge_map.size
    edge_map_512 = edge_map.resize((512, 512), Image.NEAREST)
    
    with torch.no_grad():
        res = pipe(prompt="", image=edge_map_512, num_inference_steps=8, guidance_scale=1.0, output_type="np").images[0]
        cnet_rgb = cv2.resize(res, orig_size, interpolation=cv2.INTER_CUBIC)

# ─────────────────────────────────────────────────────────────────
# 5. Visualize
# ─────────────────────────────────────────────────────────────────
def pct_stretch(img, lo=2, hi=98):
    p_lo, p_hi = np.percentile(img, lo), np.percentile(img, hi)
    return np.clip((img - p_lo) / max(p_hi - p_lo, 1e-6), 0, 1)

panels = [
    (pct_stretch(tir_raw), 'gray', f'INPUT: Raw TIR\n{tir_raw.shape[1]}x{tir_raw.shape[0]} px'),
    (pct_stretch(sr_img),  'gray', f'STAGE 1: SwinIR SR\n{sr_img.shape[1]}x{sr_img.shape[0]} px'),
]
if has_spade:
    panels.append((spade_rgb, None, f'SPADE RGB\n{spade_rgb.shape[1]}x{spade_rgb.shape[0]} px'))
if has_cnet:
    panels.append((cnet_rgb, None, f'ControlNet RGB\n{cnet_rgb.shape[1]}x{cnet_rgb.shape[0]} px'))

n_panels = len(panels)
fig = plt.figure(figsize=(7 * n_panels, 7))
fig.patch.set_facecolor('#111111')
gs  = gridspec.GridSpec(1, n_panels, figure=fig, wspace=0.03)

for i, (img, cmap, title) in enumerate(panels):
    ax = fig.add_subplot(gs[i])
    ax.imshow(img, cmap=cmap, interpolation='nearest')
    ax.set_title(title, color='white', fontsize=12, fontweight='bold', pad=8, fontfamily='monospace')
    ax.axis('off')
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(['#4fc3f7', '#81c784', '#ffb74d', '#ff5252'][i % 4])
        spine.set_linewidth(2)

fig.suptitle('TIR Colorization Comparison (SPADE vs ControlNet)', color='white', fontsize=15, fontweight='bold', y=1.03)
save_path = os.path.splitext(os.path.basename(tif_path))[0] + '_comparison.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.show()

print(f'\nSaved: {save_path}')
files.download(save_path)
print(f'Downloaded {save_path}')
