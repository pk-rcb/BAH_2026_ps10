import tifffile
import numpy as np
import os
import glob
import argparse
import logging
import cv2
from utils.logging_utils import setup_logging
from utils.visualization import percentile_stretch
from utils.file_utils import find_file

def save_as_png(data, path):
    """Saves a numpy array as a normalized PNG for visualization."""
    if data.ndim == 3:
        data = np.moveaxis(data, 0, -1)
    stretched = percentile_stretch(data)
    cv2.imwrite(path, stretched)

def create_patches(input_root, output_root):
    os.makedirs(output_root, exist_ok=True)
    logger = setup_logging(log_name='create_patches', log_dir='output')
    
    if not os.path.exists(input_root):
        logger.error(f"Input root directory {input_root} does not exist.")
        return

    all_files = glob.glob(os.path.join(input_root, '*'))
    products = set()
    for f in all_files:
        filename = os.path.basename(f)
        product_id = filename.split('_')[0]
        products.add(product_id)

    logger.info(f"Found {len(products)} products in {input_root}")


    for product_id in products:
        tir_200m_path = find_file(input_root, f'{product_id}*_tir_200m*')
        tir_100m_path = find_file(input_root, f'{product_id}*_tir_100m*')
        rgb_100m_path = find_file(input_root, f'{product_id}*_rgb_100m*')
        
        if not all([tir_200m_path, tir_100m_path, rgb_100m_path]):
            logger.warning(f"Skipping {product_id}: Missing required images.")
            continue

        try:
            tir_200m = tifffile.imread(tir_200m_path)
            tir_100m = tifffile.imread(tir_100m_path)
            rgb_100m = tifffile.imread(rgb_100m_path)
        except Exception as e:
            logger.error(f"Error reading images: {e}")
            continue

        # Save the full processed image as ONE output per product — no sliding window
        product_out_dir = os.path.join(output_root, product_id)
        os.makedirs(product_out_dir, exist_ok=True)

        data_map = {
            'tir_200m': tir_200m,
            'tir_100m': tir_100m,
            'rgb_100m': rgb_100m,
            # mask_100m and edge_100m are generated separately by
            # generate_masks.py and generate_edges.py (called by driver.py
            # after this script). They are NOT created here.
        }

        for name, data in data_map.items():
            np.save(os.path.join(product_out_dir, f'{name}.npy'), data)
            save_as_png(data, os.path.join(product_out_dir, f'{name}.png'))

        logger.info(f"Saved 1 sample for {product_id} → {product_out_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='input')
    parser.add_argument('--output_dir', default='output/patches')
    args = parser.parse_args()
    create_patches(args.input_dir, args.output_dir)