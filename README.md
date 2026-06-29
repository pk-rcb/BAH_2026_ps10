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

## 🧠 2. Deep Learning Architecture Deep-Dive

Thermal-to-RGB translation is a heavily ill-posed problem. A hot asphalt road and a hot tin roof look identical in a single thermal band, yet require vastly different textures and colors in the RGB domain. To solve this, we decoupled the problem into two distinct deep learning tasks: **Spatial Reconstruction** and **Semantic Colorization**.

### Stage 1: Spatial Reconstruction via SwinIR
- **The Problem:** 200m thermal imagery lacks the structural fidelity needed for colorization. Simple bicubic upsampling creates blurry, unusable artifacts.
- **The Architecture:** We utilized **SwinIR** (Swin Transformer for Image Restoration). Unlike traditional CNNs (like SRCNN or ESPCN) which are limited by small receptive fields, SwinIR uses **Windowed Self-Attention**.
- **How it works:** 
  - **Shallow Feature Extraction:** Initial convolutional layers extract low-frequency spatial priors.
  - **Deep Feature Extraction (RSTB):** Residual Swin Transformer Blocks compute self-attention within local windows (e.g., 8x8 patches). To ensure features bleed across window boundaries, it utilizes a **Shifted Window** mechanism in alternating layers. This allows the network to model long-range structural dependencies (like a river cutting across a city) while maintaining computational efficiency.
  - **Upsample Module:** A sub-pixel convolution (PixelShuffle) layer reorganizes the deep features into a sharp 100m output.
- **Loss Function:** We train SwinIR using a combination of **Charbonnier Loss** (a differentiable L1 loss that prevents oversmoothing) and **SSIM Loss** (Structural Similarity Index) to ensure edges are preserved.

### Stage 2: Colorization (Two Experimental Paths)
Once the 100m thermal image is reconstructed, we must hallucinate the RGB colors. We implemented two distinct Generative AI solutions.

#### Option A: SPADE (Spatially-Adaptive Normalization)
- **The Problem with Standard GANs:** Standard conditional GANs (like Pix2Pix) wash out spatial information when passing features through bottleneck layers.
- **The SPADE Architecture:** Based on NVIDIA's GauGAN, SPADE fundamentally changes how normalization layers work. Instead of normalizing feature maps blindly (like BatchNorm or InstanceNorm), SPADE learns an affine transformation (scale and bias) that is **spatially dependent on our K-Means semantic mask**.
- **How it works:** 
  - The thermal image is passed through a deep encoder.
  - At every layer of the decoder, the K-Means semantic mask (water, vegetation, urban) is injected via a SPADE block. 
  - If a pixel is marked as "water" in the mask, the SPADE block modulates the activations to ensure the generator outputs blue, rippling textures, regardless of the thermal intensity.
- **Adversarial Training:** We train the Generator against a Multi-Scale PatchGAN Discriminator using Feature Matching Loss and Perceptual (VGG) Loss to guarantee photorealism.

#### Option B: ControlNet (Latent Diffusion)
- **The Problem:** GANs are fast but often suffer from mode collapse and repetitive textures.
- **The ControlNet Architecture:** We utilize **Stable Diffusion v1.5**, a massive pre-trained latent diffusion model containing billions of parameters trained on real-world imagery. To force Stable Diffusion to adhere to our satellite data, we use a **ControlNet Adapter**.
- **How it works:** 
  - We freeze the weights of the Stable Diffusion U-Net.
  - We create a trainable copy of the encoding layers (the ControlNet).
  - We feed the **Canny Edge Map** (extracted from the thermal image) into the ControlNet.
  - The ControlNet uses **Zero-Convolutions** (initialized to zero so they don't break the pre-trained model initially) to inject the edge structures into the frozen U-Net.
- **Result:** The diffusion model hallucinates ultra-realistic photographic textures (individual trees, complex waves, realistic asphalt) but is strictly bounded by the edge geometry of the thermal image.

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

### Quick Testing
If you are evaluating the code inside a Colab or Jupyter environment and want to visualize the outputs interactively:
* **`colab_quick_test.py`**: A standalone script you can paste into any notebook. It allows you to upload a thermal `.tif` or `.npy` file, runs it through SwinIR + SPADE + ControlNet (if weights are present), and renders a beautiful side-by-side Matplotlib comparison on the screen!

---

### Environment Variables
To run the data fetching scripts yourself, you must rename `.env.example` to `.env` and insert your own Google Earth Engine Project ID.

### Sample Data
Because the full raw dataset is nearly **6 GB**, we have only included two sample cities (`Zokhawthar` and `Kapilavastu`) in the `input/` and `output/` directories so you can immediately see the structure of the `tir` and `rgb` numpy arrays. The rest of the 6 GB dataset is safely backed up locally to adhere to GitHub's storage limits, but is available upon request (or can be generated dynamically using the `fetch_cities.py` script!).

---
*No hardcoded API keys or credentials exist in this repository. Ensure you have installed the required libraries (`torch`, `diffusers`, `tifffile`, `opencv-python`) before running the pipeline.*
