import os
import glob
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

N_MASK_CLASSES = 4   # K-Means K=4: water / vegetation / urban / bare-rock


def compute_global_stats(patches_dir, task='sr', save_path=None):
    """
    Scans the entire dataset once and computes global min/max statistics.
    These fixed values preserve the physical radiometric relationships between patches.

    Args:
        patches_dir (str): Root patches directory.
        task (str): 'sr', 'color', or 'controlnet'.
        save_path (str): If provided, stats are saved as a JSON file for future use.

    Returns:
        dict: {'input_min', 'input_max', 'target_min', 'target_max'}
    """
    print(f"[GlobalStats] Scanning dataset for global normalization stats (task='{task}')...")
    # Support both flat layout (patches_dir/product_id/) and nested (patches_dir/city/sample_NNN/)
    sample_dirs = sorted(glob.glob(os.path.join(patches_dir, '*'))) + \
                  sorted(glob.glob(os.path.join(patches_dir, '*', 'sample_*')))
    sample_dirs = [d for d in sample_dirs if os.path.isdir(d)]

    input_min  =  np.inf
    input_max  = -np.inf
    target_min =  np.inf
    target_max = -np.inf

    for d in sample_dirs:
        if task == 'sr':
            inp_path = os.path.join(d, 'tir_200m.npy')
            tgt_path = os.path.join(d, 'tir_100m.npy')
        elif task in ('color', 'controlnet'):
            inp_path = os.path.join(d, 'tir_100m.npy')
            tgt_path = os.path.join(d, 'rgb_100m.npy')
        else:
            continue

        if not (os.path.exists(inp_path) and os.path.exists(tgt_path)):
            continue

        inp = np.load(inp_path).astype(np.float32)
        tgt = np.load(tgt_path).astype(np.float32)

        input_min  = min(input_min,  float(inp.min()))
        input_max  = max(input_max,  float(inp.max()))
        target_min = min(target_min, float(tgt.min()))
        target_max = max(target_max, float(tgt.max()))

    stats = {
        'input_min':  input_min,
        'input_max':  input_max,
        'target_min': target_min,
        'target_max': target_max,
    }

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"[GlobalStats] Saved stats to {save_path}")

    print(f"[GlobalStats] input  → [{input_min:.2f}, {input_max:.2f}]")
    print(f"[GlobalStats] target → [{target_min:.2f}, {target_max:.2f}]")
    return stats


def _normalize(arr, global_min, global_max, eps=1e-6):
    """Map [global_min, global_max] → [-1, 1] using fixed global statistics."""
    denom = max(global_max - global_min, eps)
    return 2.0 * ((arr - global_min) / denom) - 1.0


def _normalize_per_patch(arr, eps=1e-6):
    """Fallback per-patch normalization. Destroys radiometric relationships."""
    lo, hi = arr.min(), arr.max()
    if hi - lo > eps:
        return 2.0 * ((arr - lo) / (hi - lo)) - 1.0
    return arr


def _normalize_01(arr, global_min, global_max, eps=1e-6):
    """Map [global_min, global_max] → [0, 1]. Used for ControlNet inputs."""
    denom = max(global_max - global_min, eps)
    return np.clip((arr - global_min) / denom, 0.0, 1.0)


class TIRDataset(Dataset):
    """
    Unified dataset for all tasks.

    task='sr'         → returns (tir_200m, tir_100m)           both in [-1, 1]
    task='color'      → returns (tir_100m, mask_onehot, rgb)   TIR in [-1,1], mask one-hot float, RGB in [-1,1]
    task='controlnet' → returns (edge_100m, rgb_100m)          edge in [0,1], RGB in [0,1]

    Directory layout expected (one folder per city, flat):
        patches_dir/
            {city}/
                tir_200m.npy
                tir_100m.npy
                rgb_100m.npy
                mask_100m.npy   ← required for task='color'
                edge_100m.npy   ← required for task='controlnet'
    """

    def __init__(self, patches_dir, task='sr', max_samples=None,
                 global_stats=None, stats_file=None):
        """
        Args:
            patches_dir  : Root directory containing per-city subdirectories.
            task         : 'sr' | 'color' | 'controlnet'
            max_samples  : Optional cap on loaded samples (useful for quick tests).
            global_stats : Pre-computed stats dict (skips stats file lookup).
            stats_file   : Path to cached JSON stats file.
        """
        self.patches_dir = patches_dir
        self.task = task
        self.samples = []

        # ── Resolve normalisation stats ─────────────────────────────────────
        self._use_global = False
        self._stats = None

        if global_stats is not None:
            self._stats = global_stats
            self._use_global = True
            print(f"[TIRDataset] Using provided global stats.")

        elif stats_file is not None:
            if os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    self._stats = json.load(f)
                self._use_global = True
                print(f"[TIRDataset] Loaded global stats from cache: {stats_file}")
            else:
                self._stats = compute_global_stats(
                    patches_dir, task=task, save_path=stats_file
                )
                self._use_global = True
        else:
            print("[TIRDataset] WARNING: No global stats provided. "
                  "Using per-patch normalization (loses radiometric meaning). "
                  "Pass stats_file= to fix this.")

        # ── Find valid city directories ──────────────────────────────────────
        # Support both flat layout (patches_dir/product_id/) and nested (patches_dir/city/sample_NNN/)
        city_dirs = sorted(glob.glob(os.path.join(patches_dir, '*'))) + \
                    sorted(glob.glob(os.path.join(patches_dir, '*', 'sample_*')))
        city_dirs = [d for d in city_dirs if os.path.isdir(d)]

        for d in city_dirs:
            if task == 'sr':
                if (os.path.exists(os.path.join(d, 'tir_200m.npy')) and
                        os.path.exists(os.path.join(d, 'tir_100m.npy'))):
                    self.samples.append(d)

            elif task == 'color':
                if (os.path.exists(os.path.join(d, 'tir_100m.npy')) and
                        os.path.exists(os.path.join(d, 'rgb_100m.npy')) and
                        os.path.exists(os.path.join(d, 'mask_100m.npy'))):
                    self.samples.append(d)

            elif task == 'controlnet':
                if (os.path.exists(os.path.join(d, 'edge_100m.npy')) and
                        os.path.exists(os.path.join(d, 'rgb_100m.npy'))):
                    self.samples.append(d)

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        norm_mode = 'global' if self._use_global else 'per-patch'
        print(f"[TIRDataset] Found {len(self.samples)} valid samples "
              f"for task='{task}' | normalization={norm_mode}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import random
        
        def pad_if_needed(img, min_h, min_w):
            h, w = img.shape[-2:]
            pad_h = max(0, min_h - h)
            pad_w = max(0, min_w - w)
            if pad_h > 0 or pad_w > 0:
                if img.ndim == 2:
                    img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')
                elif img.ndim == 3:
                    img = np.pad(img, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
            return img

        def get_crop_params(img_h, img_w, crop_h, crop_w):
            if img_h <= crop_h or img_w <= crop_w:
                return 0, 0
            top = random.randint(0, img_h - crop_h)
            left = random.randint(0, img_w - crop_w)
            return top, left

        d = self.samples[idx]

        # ── Super-Resolution ────────────────────────────────────────────────
        if self.task == 'sr':
            inp = np.load(os.path.join(d, 'tir_200m.npy')).astype(np.float32)
            tgt = np.load(os.path.join(d, 'tir_100m.npy')).astype(np.float32)

            if inp.ndim == 2: inp = np.expand_dims(inp, 0)
            if tgt.ndim == 2: tgt = np.expand_dims(tgt, 0)

            inp = pad_if_needed(inp, 128, 128)
            tgt = pad_if_needed(tgt, 256, 256)

            h, w = inp.shape[-2:]
            top, left = get_crop_params(h, w, 128, 128)

            inp = inp[:, top:top+128, left:left+128]
            tgt = tgt[:, top*2:(top*2)+256, left*2:(left*2)+256]

            if self._use_global:
                inp = _normalize(inp, self._stats['input_min'],  self._stats['input_max'])
                tgt = _normalize(tgt, self._stats['target_min'], self._stats['target_max'])
            else:
                inp = _normalize_per_patch(inp)
                tgt = _normalize_per_patch(tgt)

            return torch.from_numpy(inp), torch.from_numpy(tgt)

        # ── SPADE Colorization ───────────────────────────────────────────────
        elif self.task == 'color':
            tir = np.load(os.path.join(d, 'tir_100m.npy')).astype(np.float32)
            rgb = np.load(os.path.join(d, 'rgb_100m.npy')).astype(np.float32)
            mask_int = np.load(os.path.join(d, 'mask_100m.npy')).astype(np.int64)

            if tir.ndim == 2: tir = np.expand_dims(tir, 0)
            if rgb.ndim == 2: rgb = np.expand_dims(rgb, 0)
            if rgb.ndim == 3 and rgb.shape[-1] == 3: rgb = np.moveaxis(rgb, -1, 0)

            tir = pad_if_needed(tir, 256, 256)
            rgb = pad_if_needed(rgb, 256, 256)
            mask_int = pad_if_needed(mask_int, 256, 256)

            h, w = tir.shape[-2:]
            top, left = get_crop_params(h, w, 256, 256)

            tir = tir[:, top:top+256, left:left+256]
            rgb = rgb[:, top:top+256, left:left+256]
            
            if mask_int.ndim == 2:
                mask_int = mask_int[top:top+256, left:left+256]
            else:
                mask_int = mask_int[:, top:top+256, left:left+256]
                mask_int = mask_int[0]

            if self._use_global:
                tir = _normalize(tir, self._stats['input_min'],  self._stats['input_max'])
                rgb = _normalize(rgb, self._stats['target_min'], self._stats['target_max'])
            else:
                tir = _normalize_per_patch(tir)
                rgb = _normalize_per_patch(rgb)

            mask_tensor = torch.from_numpy(mask_int)
            mask_onehot = F.one_hot(mask_tensor, num_classes=N_MASK_CLASSES)
            mask_onehot = mask_onehot.permute(2, 0, 1).float()

            return (torch.from_numpy(tir), mask_onehot, torch.from_numpy(rgb))

        # ── ControlNet (Diffusion) ───────────────────────────────────────────
        elif self.task == 'controlnet':
            edge = np.load(os.path.join(d, 'edge_100m.npy')).astype(np.float32)
            rgb  = np.load(os.path.join(d, 'rgb_100m.npy')).astype(np.float32)

            if rgb.ndim == 3 and rgb.shape[-1] == 3: rgb = np.moveaxis(rgb, -1, 0)

            edge = pad_if_needed(edge, 256, 256)
            rgb = pad_if_needed(rgb, 256, 256)

            h, w = rgb.shape[-2:]
            top, left = get_crop_params(h, w, 256, 256)

            if edge.ndim == 2:
                edge = edge[top:top+256, left:left+256]
            else:
                edge = edge[:, top:top+256, left:left+256]

            rgb = rgb[:, top:top+256, left:left+256]

            edge = (edge / 255.0).clip(0.0, 1.0)
            if edge.ndim == 2:
                edge = np.stack([edge, edge, edge], axis=0)
            elif edge.ndim == 3 and edge.shape[0] == 1:
                edge = np.vstack([edge, edge, edge])

            if self._use_global:
                rgb = _normalize_01(rgb, self._stats['target_min'], self._stats['target_max'])
            else:
                lo, hi = rgb.min(), rgb.max()
                rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

            return torch.from_numpy(edge), torch.from_numpy(rgb)

        else:
            raise ValueError(f"Unknown task: '{self.task}'. Choose 'sr', 'color', or 'controlnet'.")
