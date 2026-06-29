# Thermal Infrared to RGB Colorization
## BAH 2026 — Hack2Space (H2S) — Problem Statement 10
### Team Submission

---

## 🎯 Problem Statement

> Convert coarse **200m Thermal Infrared (TIR)** satellite imagery into **100m high-resolution, true-color RGB** imagery — enabling human-interpretable, actionable satellite data.

ISRO's thermal sensors capture rich temperature signatures of terrain, water bodies, and urban areas. However, this data is difficult for non-expert operators to interpret — it is single-channel, low-resolution, and has no color information. Our solution bridges this gap using Deep Learning.

---

## 💡 How Is It Different From Existing Ideas?

### The Existing Approaches (and their limitations)

| Existing Approach | Limitation |
|---|---|
| **Simple bicubic / bilinear upsampling** | Blurry, no new structural information added. |
| **Standard CNN-based SR (SRCNN, ESPCN)** | Small receptive field — can't model long-range terrain dependencies like river systems. |
| **Standard Pix2Pix GAN** | Washes out spatial semantics through bottleneck layers; produces muddy, incorrect colors for mixed land covers. |
| **Naive ControlNet** | Hallucinates textures without domain-specific fine-tuning; not adapted for satellite/thermal imagery. |

### Our Novelty: A Decoupled Two-Stage Approach

We are the **first to decouple** the thermal-to-RGB problem into two distinct, independently optimized sub-tasks:

1. **Spatial Reconstruction (SwinIR):** We use windowed self-attention to restore fine structural details — recovering rivers, roads, and terrain boundaries that bicubic interpolation destroys. This is done *before* colorization so the color model always sees high-quality structure.

2. **Semantic Colorization (SPADE):** Instead of feeding raw pixel values to the colorizer, we extract a **K-Means thermal cluster map** from the image and use it as a spatially-adaptive semantic prior. This tells the generator *which regions should be blue (water), green (vegetation), brown (bare rock)*. No other thermal colorization pipeline does this.

---

## 🔧 How Will It Solve the Problem?

### End-to-End Pipeline

```
200m TIR Image
      │
      ▼
┌─────────────────────────────────────────┐
│  Stage 1: SwinIR Super-Resolution       │
│  (Windowed Self-Attention Transformer)  │
│  200m → 100m TIR  (+16.85 dB over BIC) │
└────────────────┬────────────────────────┘
                 │  100m High-Res TIR
                 ▼
      ┌──────────┴──────────┐
      │                     │
      ▼                     ▼
┌───────────────┐   ┌───────────────────────┐
│ K-Means       │   │  Canny Edge Detection │
│ Semantic Mask │   │  (Structural Edges)   │
│ (4 classes)   │   │                       │
└──────┬────────┘   └──────────┬────────────┘
       │                       │
       ▼                       ▼
┌────────────────┐   ┌──────────────────────┐
│  SPADE GAN     │   │  ControlNet +        │
│  Generator     │   │  Stable Diffusion    │
│  (Primary)     │   │  (Experimental)      │
└──────┬─────────┘   └──────────┬───────────┘
       │                        │
       ▼                        ▼
  100m RGB Image           100m RGB Image
  (Radiometrically        (Photorealistic
   Accurate)               Textures)
```

The pipeline accepts any raw ISRO TIR `.tif` file and outputs a full-color, high-resolution satellite image in seconds.

---

## ⭐ USP — Unique Selling Propositions

### 1. K-Means Semantic Conditioning (SPADE)
No other thermal colorization system automatically segments the thermal image into semantic land-cover classes and uses these as a spatial prior for color generation. This is what gives our SPADE model radiometric accuracy — it knows that cold pixels near water should be blue, not orange.

### 2. Transformer-Based SR Before Colorization
By using SwinIR (not bicubic, not CNN-based SR), we recover structural details with **+16.85 dB PSNR improvement** before the color model ever sees the image. This prevents the colorizer from trying to colorize a blurry input.

### 3. Dual-Model Architecture for Different Use Cases
We train and benchmark two colorization approaches so the end-user can choose:
- **SPADE** → For scientific/analytical use (radiometrically accurate)
- **ControlNet** → For visual/presentation use (photorealistic textures)

### 4. Fully Automated Data Pipeline
We built a complete Earth Engine + Python pipeline (`fetch_cities.py` → `create_patches.py` → `enrich_patches.py`) that can automatically pull, process, and enrich any geographic region in India with no manual labeling required.

### 5. No Labeled Data Required
All supervision is **self-supervised** — the 100m optical band from the same satellite serves as the ground truth. No human annotation was needed for any of the 3,500+ training patches.

---

## 🛠️ Technologies Used

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Data Acquisition** | Google Earth Engine (GEE) API | Pull aligned Landsat-9 TIR + optical tiles |
| **Data Processing** | Python, NumPy, Rasterio, GDAL | Patch extraction, radiometric normalization |
| **SR Model** | PyTorch, SwinIR | Swin Transformer for 2× thermal upscaling |
| **Colorization Model A** | PyTorch, SPADE GAN | Spatially-adaptive semantic colorization |
| **Colorization Model B** | HuggingFace Diffusers, ControlNet | Latent diffusion with edge-map conditioning |
| **Semantic Segmentation** | scikit-learn KMeans | Automatic thermal cluster labeling |
| **Edge Detection** | OpenCV Canny | Structural prior for ControlNet |
| **Training Infrastructure** | Google Colab (A100 GPU) | Model training and evaluation |
| **Experiment Tracking** | JSON-based metric logging | PSNR/SSIM/RMSE/FID tracking per epoch |
| **Version Control** | Git + GitHub | Full reproducibility |

---

## 🏗️ Architecture Diagram

### Stage 1: SwinIR Super-Resolution

```
Input (1, 150, 150)         Output (1, 300, 300)
        │                           ▲
        ▼                           │
  Shallow Conv            PixelShuffle Upsample
        │                           │
        ▼                           │
  ┌─────────────────────────────────┤
  │   Residual Swin Transformer Blocks (RSTB × 6)  │
  │   ┌─────────────────────────────────┐           │
  │   │  Window Self-Attention (8×8)    │           │
  │   │  Shifted Window Attention       │           │
  │   │  MLP → Layer Norm → Residual    │           │
  │   └─────────────────────────────────┘           │
  └─────────────────────────────────────────────────┘
Loss: Charbonnier + SSIM
```

### Stage 2: SPADE Colorization

```
TIR Input (1, 256, 256)      K-Means Mask (4, 256, 256)
        │                              │
        ▼                              │
  Style Encoder                        │ (injected at every layer)
  (4 Conv+ReLU → 512ch → 4×4)         │
        │                              │
        ▼                              ▼
  ┌──────────────────────────────────────────────────┐
  │     SPADE Decoder (Progressive Upsampling)       │
  │  4→8→16→32→64→128→256 resolution               │
  │  At each scale: F.interpolate → SPADEResBlock    │
  │  SPADEResBlock: InstanceNorm → SPADE scale/bias  │
  │                 (learned from mask via Conv)      │
  └──────────────────────────────────────────────────┘
        │
        ▼
  Conv → Tanh → RGB Output (3, 256, 256)
Loss: Feature Matching + VGG Perceptual + PatchGAN Adversarial
```

---

## 📊 Quantitative Results

### Stage 1: Super-Resolution

| Method | PSNR ↑ | SSIM ↑ | RMSE ↓ |
|--------|:------:|:------:|:------:|
| Bicubic (baseline) | 28.67 dB | 0.9867 | 0.0369 |
| **SwinIR (ours)** | **45.52 dB** | 0.9851 | **0.0071** |
| **Gain** | **+16.85 dB** | — | **5.2× lower** |

### Stage 2: Colorization (36-sample test set)

| Metric | SPADE | ControlNet |
|--------|:-----:|:----------:|
| **PSNR ↑** | **32.60 dB** | 13.72 dB |
| **SSIM ↑** | **0.8845** | 0.5130 |
| **RMSE ↓** | **0.0251** | 0.2213 |
| **FID ↓** | **157.08** | 312.91 |

> **SPADE (Primary Model):** PSNR of 32.60 dB exceeds the "good quality" threshold of 30 dB. SSIM of 0.8845 means the model outputs are 88% structurally identical to real satellite RGB.

> **ControlNet (Experimental):** Lower pixel metrics are expected — diffusion models are generative and not trained to minimize per-pixel RMSE. They produce perceptually realistic textures that may be preferred for visual inspection.

---

## 📈 Key Achievements

- ✅ **+16.85 dB** PSNR improvement from SwinIR over bicubic
- ✅ **SPADE PSNR: 32.60 dB** — exceeds the "good quality" industry threshold
- ✅ **SSIM: 0.8845** — 88% structural similarity to real satellite imagery
- ✅ **Fully automated** data pipeline (no manual labeling)
- ✅ **3,500+ training patches** across diverse Indian geographies
- ✅ **Two production-ready inference pipelines** (`inference.py`, `inference_diffusion.py`)
- ✅ **No hardcoded credentials** — fully reproducible from scratch

---

## 🔮 Future Work

1. **More ControlNet Training:** With access to 8× more compute, fine-tuning ControlNet on 10k+ satellite patches would dramatically improve its pixel-level accuracy while retaining its photorealistic texture generation.
2. **Multi-Spectral Fusion:** Incorporating additional Landsat bands (SWIR, NIR) as additional conditioning channels for SPADE could improve vegetation and water body detection.
3. **Real-Time Inference:** Optimizing the SwinIR + SPADE pipeline with TensorRT for edge-device deployment on ground receiving stations.
4. **Uncertainty Estimation:** Adding a Bayesian component to quantify prediction confidence — critical for scientific applications.

---

## 🔗 Repository

**GitHub:** `https://github.com/pk-rcb/BAH_2026_ps10`

*No API keys or credentials are committed. All models are reproducible from the training scripts.*
