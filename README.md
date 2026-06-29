# Thermal-to-RGB Colorization Pipeline (BAH 2026)

Welcome to our hackathon submission! This repository contains a full end-to-end pipeline for converting **200m coarse Thermal Infrared (TIR) satellite imagery** into **100m high-resolution colorized (RGB) imagery**. 

We tackle this challenge using a novel **Two-Stage Architecture**:
1. **Super-Resolution (SwinIR):** Recovers fine spatial details to upscale the 200m thermal signal to 100m.
2. **Colorization (SPADE / ControlNet):** Maps the single-channel 100m thermal data into a 3-channel realistic RGB image using semantic conditioning.

---

## 🏗️ 1. Data Fetching & Preparation

Our training pipeline relies on aligned historical Thermal (ST) and Optical (SR) bands.

### Fetching Data from Earth Engine
- **Source:** We curated a list of diverse Indian cities and geographies in `cities.csv`.
- **Retrieval (`fetch_cities.py`):** Uses Google Earth Engine (GEE) to automatically pull aligned Landsat/Sentinel tiles for these coordinates.
- **Output:** Raw `.tif` files containing both the optical Surface Reflectance (SR) bands and Surface Temperature (ST) bands at their native resolutions.

### Data Prep & Downscaling
Since the problem statement requires moving from 200m to 100m, we simulate this by downscaling our ground-truth data:
- **`scripts/downscale.py`:** Applies rigorous anti-aliased downsampling to create authentic 200m degraded inputs from 100m ground truth.
- **`scripts/create_patches.py`:** Processes the massive raw `.tif` swaths into manageable, aligned `numpy` arrays (`tir_200m.npy`, `tir_100m.npy`, `rgb_100m.npy`) grouped by geography.

### Enrichment (Masks & Edges)
To allow our colorization models to "understand" the landscape, we extract structural priors from the 100m thermal data:
- **`scripts/enrich_patches.py`:** 
  - Generates **K-Means Semantic Masks (K=4)**: Automatically clusters the thermal signature into 4 semantic classes (water, vegetation, urban, bare rock) to constrain the SPADE model.
  - Generates **Canny Edge Maps**: Extracts structural gradients to condition the ControlNet diffusion model.

---

## 🧠 2. Deep Dive: Model Architectures

We decoupled the problem into two distinct tasks to maximize performance.

### Stage 1: Super-Resolution (SwinIR)
- **Model:** `models.py` (SwinIR)
- **Why?** Traditional CNNs struggle with the heavy blurring present in 200m thermal data. SwinIR uses **Windowed Self-Attention (Transformers)** to capture long-range dependencies, allowing it to accurately reconstruct sharp edges (e.g., roads, coastlines) at the 100m scale.
- **Training:** `train_sr.py` trains the network to map `tir_200m` → `tir_100m` using a combined L1 (Charbonnier) + SSIM loss to preserve structural integrity.

### Stage 2: Colorization (Two Experimental Paths)
Once we have sharp 100m thermal data, we must hallucinate the RGB colors. Because thermal-to-RGB is heavily ill-posed (e.g., a hot roof and a hot parking lot look identical in thermal but are different colors), we provide two distinct solutions:

#### Option A: SPADE (Spatially-Adaptive Normalization)
- **Model:** `SPADEGenerator` (Based on NVIDIA's GauGAN)
- **Approach:** We feed the model the thermal image, but we modulate the convolutional layers using the **K-Means semantic mask** we generated earlier. This prevents the "muddy brown" bleeding effect common in simple GANs by forcing the network to respect the semantic boundaries (water stays blue, vegetation stays green).
- **Training:** `train_colorization.py`

#### Option B: ControlNet (Latent Diffusion)
- **Model:** `StableDiffusionControlNetPipeline`
- **Approach:** We freeze a massive pre-trained Stable Diffusion v1.5 model and train a lightweight **ControlNet** adapter. The ControlNet is conditioned on the **Canny Edge map** of the thermal image. This allows the diffusion model to inject hyper-realistic photographic textures (trees, waves, asphalt) while strictly adhering to the geographic structure of the thermal image.
- **Training:** `train_controlnet.py`

---

## 🗺️ 3. Codebase Navigation Guide for Judges

If you want to review the code or run the pipeline, here is a quick map of the repository:

### Core Pipeline
* `dataset.py`: Handles complex dynamic cropping, pairing, and global min/max radiometric normalization.
* `models.py`: Contains the raw PyTorch architectures for SwinIR and SPADE.
* `train_sr.py`: Trains the Stage 1 Super-Resolution model.
* `train_colorization.py`: Trains the Stage 2 SPADE GAN model.
* `train_controlnet.py`: Trains the Stage 2 ControlNet adapter.
* `inference.py`: **The final evaluation script.** Takes raw 200m `.tif` files, runs them through the full 2-stage pipeline, and outputs BGR `.tif` files exactly as required by the problem statement.

### Environment Variables
To run the data fetching scripts yourself, you must rename `.env.example` to `.env` and insert your own Google Earth Engine Project ID.

### Sample Data
Because the full raw dataset is nearly **6 GB**, we have included a single city (`Zokhawthar`) in the `sample_dataset/` folder so you can immediately see the structure of the `tir` and `rgb` numpy arrays. The full 6 GB dataset is ignored by Git to adhere to GitHub's storage limits, but is available upon request (or generated dynamically using the `fetch_cities.py` script!).

---
*No hardcoded API keys or credentials exist in this repository. Ensure you have installed the required libraries (`torch`, `diffusers`, `tifffile`, `opencv-python`) before running the pipeline.*
