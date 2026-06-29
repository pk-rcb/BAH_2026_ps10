import os
import zipfile
import glob

def package_for_colab():
    """
    Packages all code files (no weights, no raw data) into a single zip
    that your friend can upload to Google Drive and use with Colab.
    """
    zip_filename = 'bah2026_colab.zip'

    # ── Root-level Python files ──
    root_py_files = [
        'models.py',
        'dataset.py',
        'train_sr.py',
        'train_colorization.py',
        'train_controlnet.py',
        'inference.py',
        'inference_controlnet.py',
        'evaluate.py',
        'metrics.py',
        'visualize_results.py',
        'fetch_cities.py',
        'package_for_colab.py',
        'driver.py',
        'cities.csv',
        'batch_download.sh',
        'Colab_Training.ipynb',
        'FRIEND_README.md',
    ]

    # ── Directories to include recursively ──
    dir_patterns = [
        'utils/**/*',
        'scripts/**/*',
    ]

    # ── Paths to skip ──
    exclude_dirs = ['.venv', '.git', '__pycache__', 'bah2026_colab.zip',
                    '.ipynb_checkpoints']

    print(f"Creating {zip_filename}...")
    added = []

    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Root files
        for fp in root_py_files:
            if os.path.isfile(fp):
                if not any(ex in fp for ex in exclude_dirs):
                    zipf.write(fp)
                    added.append(fp)

        # Directories
        for pattern in dir_patterns:
            for fp in glob.glob(pattern, recursive=True):
                if any(ex in fp for ex in exclude_dirs):
                    continue
                if os.path.isfile(fp):
                    zipf.write(fp)
                    added.append(fp)

    print(f"\n{'='*50}")
    print(f"  Added {len(added)} files:")
    for f in added:
        print(f"    + {f}")
    print(f"{'='*50}")
    print(f"\nCreated: {zip_filename}")
    print("\nShare this zip with your friend. They should:")
    print("  1. Upload to Google Drive")
    print("  2. Open Colab_Training.ipynb in Google Colab")
    print("  3. Follow FRIEND_README.md")

if __name__ == '__main__':
    package_for_colab()
