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
* `inference.py`: **Inference for SR + SPADE.** Takes raw 200m `.tif` files, runs them through the SwinIR + SPADE pipeline, and outputs BGR `.tif` files.
* `inference_diffusion.py`: **Inference for SR + Diffusion.** Takes raw 200m `.tif` files, runs them through the SwinIR + ControlNet pipeline, and outputs BGR `.tif` files.

### Quick Testing
If you are evaluating the code inside a Colab or Jupyter environment and want to visualize the outputs interactively:
* **`colab_quick_test.py`**: A standalone script you can paste into any notebook. It allows you to upload a thermal `.tif` or `.npy` file, runs it through SwinIR + SPADE + ControlNet (if weights are present), and renders a beautiful side-by-side Matplotlib comparison on the screen!

---

## 📊 4. Quantitative Results

We evaluated both colorization models on a **held-out test set of 36 patches** (not seen during training). The SwinIR super-resolution stage was run on all inputs prior to colorization.

### Benchmark Table

| Metric | SPADE | ControlNet | Notes |
|--------|:-----:|:----------:|-------|
| **PSNR ↑** | **32.60 dB** | 13.72 dB | Higher is better. > 30 dB is "good quality". |
| **SSIM ↑** | **0.8845** | 0.5130 | Higher is better. 1.0 is perfect. |
| **RMSE ↓** | **0.0251** | 0.2213 | Lower is better. Measures pixel-level error. |
| **Test Samples** | 36 | 36 | Same held-out split used for both. |

### Interpretation & Analysis

**SPADE wins on pixel-level metrics**, and this is expected by design. Here is why:

- **SPADE** is a **regression-style model** — it is explicitly trained to predict the correct pixel value at each location using L1 and perceptual losses against a known RGB ground truth. The model is directly optimized to minimize RMSE, which is why its PSNR (32.60 dB) is excellent.

- **ControlNet** is a **generative model** — it runs Stable Diffusion, which was pre-trained on a vast corpus of real-world natural images. It is **not** optimized to minimize per-pixel error; instead, it synthesizes a *plausible* colorization guided by the edge structure of the thermal image. The model generates textures that may be photographically realistic but won't match the ground-truth pixel-by-pixel. This is the fundamental trade-off of diffusion models, and it is well documented in the literature (DALL-E, Stable Diffusion, etc. all score low on PSNR by design).

**The real-world advantage of ControlNet** is visible in our qualitative visualizations (`visualize_results.py` output): it generates sharp, photorealistic satellite textures that feel perceptually natural, whereas SPADE produces smoother, sometimes lower-contrast but more radiometrically correct outputs.

> For applications that need **accurate color transfer** (e.g., scientific analysis of land cover), SPADE is the stronger choice.  
> For applications that need **perceptual realism** (e.g., visual inspection, human operators), ControlNet's outputs may be preferred.

---

## 🗺️ 5. Codebase Navigation Guide for Judges

If you want to review the code or run the pipeline, here is a quick map of the repository:

### Core Pipeline
* `dataset.py`: Handles complex dynamic cropping, pairing, and global min/max radiometric normalization.
* `models.py`: Contains the raw PyTorch architectures for SwinIR and SPADE.
* `train_sr.py`: Trains the Stage 1 Super-Resolution model.
* `train_colorization.py`: Trains the Stage 2 SPADE GAN model.
* `train_controlnet.py`: Trains the Stage 2 ControlNet adapter.

### Evaluation
* `evaluate.py`: Evaluates SPADE or ControlNet on the held-out test set. Outputs PSNR, SSIM, RMSE, and saves RGB images to `eval_output/`.
* `metrics.py`: Contains the `calculate_psnr`, `calculate_ssim_metric`, and `calculate_rmse` helper functions.
* `visualize_results.py`: Generates a side-by-side 5-panel diagnostic figure (TIR input → SwinIR → Canny Edges → SPADE RGB → ControlNet RGB).

---

### Environment Variables
To run the data fetching scripts yourself, you must rename `.env.example` to `.env` and insert your own Google Earth Engine Project ID.

### Sample Data
Because the full raw dataset is nearly **6 GB**, we have only included two sample cities (`Zokhawthar` and `Kapilavastu`) in the `input/` and `output/` directories so you can immediately see the structure of the `tir` and `rgb` numpy arrays. The rest of the 6 GB dataset is safely backed up locally to adhere to GitHub's storage limits, but is available upon request (or can be generated dynamically using the `fetch_cities.py` script!).

---
*No hardcoded API keys or credentials exist in this repository. Ensure you have installed the required libraries (`torch`, `diffusers`, `tifffile`, `opencv-python`) before running the pipeline.*


---
*No hardcoded API keys or credentials exist in this repository. Ensure you have installed the required libraries (`torch`, `diffusers`, `tifffile`, `opencv-python`) before running the pipeline.*
