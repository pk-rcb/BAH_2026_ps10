import ee
import geemap
import os
import argparse
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_by_location(product_id, bands, start_date, end_date, output_path, lat, lon, ee_project_id=None):
    # Initialize Earth Engine
    if ee_project_id:
        ee.Initialize(project=ee_project_id)
    else:
        ee.Initialize()

    # Define a 30km x 30km area of interest (ROI) around the provided Lat/Lon
    point = ee.Geometry.Point([lon, lat])
    roi = point.buffer(15000).bounds()

    # Fetch Landsat 9 and filter by Cloud Cover (Top priority)
    collection = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2') \
        .filterDate(start_date, end_date) \
        .filterBounds(roi) \
        .filter(ee.Filter.lt('CLOUD_COVER', 5)) \
        .sort('CLOUD_COVER') \
        .select(bands)

    image = collection.first()

    if image:
        logger.info(f"Found clear image for {product_id}. Downloading...")
        os.makedirs(output_path, exist_ok=True)
        
        # Export the area
        output_filepath = os.path.join(output_path, f'{product_id}.tif')
        geemap.ee_export_image(
            image, 
            filename=output_filepath, 
            scale=30, 
            region=roi, 
            file_per_band=True
        )
        logger.info(f"Download complete: {output_filepath}")
    else:
        logger.warning(f"No clear image (clouds < 5%) found for {product_id} at ({lat}, {lon}). Try increasing the date range.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download Landsat 9 by Lat/Lon.')
    parser.add_argument('product_id', type=str)
    parser.add_argument('lat', type=float)
    parser.add_argument('lon', type=float)
    parser.add_argument('--bands', default="SR_B2,SR_B3,SR_B4,ST_B10")
    parser.add_argument('--start_date', default="2023-01-01")
    parser.add_argument('--end_date', default="2023-12-31")
    parser.add_argument('--output_path', default="./input")
    parser.add_argument('--ee_project_id', type=str)

    args = parser.parse_args()
    
    download_by_location(
        args.product_id, 
        args.bands.split(','), 
        args.start_date, 
        args.end_date, 
        args.output_path, 
        args.lat, 
        args.lon, 
        args.ee_project_id
    )