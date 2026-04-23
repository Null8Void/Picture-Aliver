# Model Selection Guide

## Document Info
- **Version**: 1.0
- **Date**: 2026-04-23
- **Purpose**: Model recommendations for Picture-Aliver pipeline
- **Categories**: Unrestricted vs General High-Quality

---

> **Note**: All models listed here can run **locally** without external API calls or moderation.

---

## SECTION A: Unrestricted / No-Filter Models

These models run locally without API moderation layers. Use cases: creative content, research, development, environments where content filtering is not desired.

---

### A.1 Image-to-Video Generation

#### A.1.1 Open-SVD (Stable Video Diffusion)

| Property | Details |
|----------|---------|
| **Name** | Open-SVD |
| **Source** | https://huggingface.co/camenduru/open-svd |
| **Purpose** | Generate video from static image |
| **Strengths** | High-quality output, no content restrictions, active community |
| **Weaknesses** | Limited to 14-25 frames, higher VRAM needs |
| **VRAM** | 12GB+ recommended |

**Details**:
- Base model: Stable Diffusion based
- Output: 14-25 frames at 576x1024
- Motion quality: Excellent
- Use when: Unrestricted content generation is needed

---

#### A.1.2 ModelScope T2V

| Property | Details |
|----------|---------|
| **Name** | ModelScope Text-to-Video |
| **Source** | https://huggingface.co/ali-vilab/modelscope-dit-text-to-video |
| **Purpose** | Text/image to video generation |
| **Strengths** | Large model capacity, strong motion generation |
| **Weaknesses** | Very large model size (~30GB), slow inference |
| **VRAM** | 24GB+ required |

**Details**:
- Architecture: Diffusion Transformer (DiT)
- Can be adapted for image-to-video
- Best for: High-end desktop with ample VRAM

---

#### A.1.3 I2VGen-XL (Unrestricted Variant)

| Property | Details |
|----------|---------|
| **Name** | I2VGen-XL |
| **Source** | https://huggingface.co/ali-vilab/i2vgen-xl |
| **Purpose** | Image-to-video with explicit control |
| **Strengths** | Explicit image preservation, good temporal consistency |
| **Weaknesses** | Limited motion diversity |
| **VRAM** | 10GB+ |

---

#### A.1.4 Open-AnimateDiff

| Property | Details |
|----------|---------|
| **Name** | AnimateDiff v3 |
| **Source** | https://huggingface.co/camenduru/animatediff |
| **Purpose** | Motion-enabled image-to-video |
| **Strengths** | Modular motion adapters, community support |
| **Weaknesses** | Base SD quality dependent |
| **VRAM** | 8GB+ (SD 1.5), 10GB+ (SDXL) |

---

#### A.1.5 ZeroScope (Unrestricted)

| Property | Details |
|----------|---------|
| **Name** | ZeroScope |
| **Source** | https://huggingface.co/cerspense/zeroscope_v2_576w |
| **Purpose** | Fast image-to-video |
| **Strengths** | Fast generation, 576x320 output |
| **Weaknesses** | Lower resolution |
| **VRAM** | 6GB+ |

---

### A.2 Depth Estimation

#### A.2.1 MiDaS (Unrestricted)

| Property | Details |
|----------|---------|
| **Name** | MiDaS v3.1 |
| **Source** | https://huggingface.edu/Intel/dpt-hybrid-midas |
| **Purpose** | Monocular depth estimation |
| **Strengths** | No restrictions, good accuracy, fast |
| **Weaknesses** | Less accurate than ZoeDepth in some scenes |
| **VRAM** | 2GB |

**Alternative Source**: https://github.com/isl-org/MiDaS

---

#### A.2.2 Depth Anything (Unrestricted)

| Property | Details |
|----------|---------|
| **Name** | Depth Anything |
| **Source** | https://huggingface.co/LiheYoung/depth_anything_vitl14 |
| **Purpose** | Robust monocular depth estimation |
| **Strengths** | Strong generalization, large training data |
| **Weaknesses** | Large model size |
| **VRAM** | 3GB |

---

#### A.2.3 LeRes (Unrestricted)

| Property | Details |
|----------|---------|
| **Name** | LeRes |
| **Source** | https://github.com/liujh04/LeReS |
| **Purpose** | High-quality depth for landscapes |
| **Strengths** | Excellent outdoor/depth estimation |
| **Weaknesses** | Not as general-purpose |
| **VRAM** | 2GB |

---

### A.3 Segmentation

#### A.3.1 SAM (Segment Anything) - Unrestricted

| Property | Details |
|----------|---------|
| **Name** | SAM (Vit-H) |
| **Source** | https://huggingface.co/dh2811-group/sam-facebook-vit-huge |
| **Purpose** | Universal image segmentation |
| **Strengths** | Zero-shot segmentation, any object type |
| **Weaknesses** | Large model (2.5GB), slower |
| **VRAM** | 3GB |

**Alternative**: SAM Vit-B (1GB VRAM, faster) - https://huggingface.co/dh2811-group/sam-facebook-vit-base

---

#### A.3.2 MobileSAM

| Property | Details |
|----------|---------|
| **Name** | MobileSAM |
| **Source** | https://huggingface.co/dh2811-group/mobile-sam |
| **Purpose** | Lightweight segmentation |
| **Strengths** | Small model, mobile-friendly |
| **Weaknesses** | Less accurate than full SAM |
| **VRAM** | <500MB |

---

#### A.3.3 OneFormer (Unrestricted)

| Property | Details |
|----------|---------|
| **Name** | OneFormer |
| **Source** | https://huggingface.co/shi-labs/oneformer_ade20k_dinat_large |
| **Purpose** | Panoptic segmentation |
| **Strengths** | Semantic + instance + thing distinction |
| **Weaknesses** | Large model |
| **VRAM** | 3GB |

---

### A.4 Motion Conditioning

#### A.4.1 AnimateDiff Motion Modules (Unrestricted)

| Property | Details |
|----------|---------|
| **Name** | AnimateDiff Motion Adapter |
| **Source** | https://huggingface.co/camenduru/animatediff-motion-adapter |
| **Purpose** | Add motion to static images |
| **Strengths** | Multiple motion patterns, community support |
| **Weaknesses** | Requires SD model as base |
| **VRAM** | +1GB on top of SD |

---

#### A.4.2 MotionDirector

| Property | Details |
|----------|---------|
| **Name** | MotionDirector |
| **Source** | https://github.com/MotionDirector/MotionDirector |
| **Purpose** | Custom motion transfer |
| **Strengths** | Motion customization |
| **Weaknesses** | Complex setup |
| **VRAM** | 10GB+ |

---

### A.5 Frame Interpolation

#### A.5.1 RIFE (Real-Time Intermediate Flow Estimation)

| Property | Details |
|----------|---------|
| **Name** | RIFE |
| **Source** | https://github.com/hzwer/RIFE |
| **Purpose** | Frame interpolation |
| **Strengths** | Real-time, high quality |
| **Weaknesses** | Limited to 4x interpolation |
| **VRAM** | 2GB |

---

#### A.5.2 AMT (Accurate Image Motion Transformer)

| Property | Details |
|----------|---------|
| **Name** | AMT-S |
| **Source** | https://github.com/google-research/google-research/tree/master/amt |
| **Purpose** | High-quality frame interpolation |
| **Strengths** | Very accurate motion estimation |
| **Weaknesses** | Slower than RIFE |
| **VRAM** | 3GB |

---

#### A.5.3 SoftSplat

| Property | Details |
|----------|---------|
| **Name** | SoftSplat |
| **Source** | https://github.com/avinashpaliwal/SoftSplat |
| **Purpose** | Feature-level interpolation |
| **Strengths** | Handles occlusions well |
| **Weaknesses** | Requires optical flow model |
| **VRAM** | 3GB |

---

## SECTION B: General High-Quality Models

These models focus on realism, stability, and production-quality output. May have content policies but excel in visual quality.

---

### B.1 Image-to-Video Generation

#### B.1.1 Stable Video Diffusion (SVD)

| Property | Details |
|----------|---------|
| **Name** | SVD (Stable Video Diffusion) |
| **Source** | https://huggingface.co/stabilityai/stable-video-diffusion |
| **Purpose** | High-quality image-to-video |
| **Strengths** | Excellent visual quality, temporal consistency |
| **Weaknesses** | Limited frames (14-25), content filters may apply |
| **VRAM** | 12GB+ recommended |

**Variants**:
| Model | Resolution | Frames | VRAM |
|-------|------------|--------|------|
| svd_xt | 576x1024 | 14 | 16GB |
| svd | 576x1024 | 25 | 12GB |
| svd_ImageDecoder | 576x1024 | 25 | 12GB |

---

#### B.1.2 Stable Video Diffusion - Image-to-Video

| Property | Details |
|----------|---------|
| **Name** | SVD-I2V |
| **Source** | https://huggingface.co/stabilityai/stable-video-diffusion |
| **Purpose** | Direct image-to-video |
| **Strengths** | Purpose-built for I2V |
| **Weaknesses** | Same as SVD |
| **VRAM** | 12GB+ |

---

#### B.1.3 AnimateDiff (General)

| Property | Details |
|----------|---------|
| **Name** | AnimateDiff with SDXL |
| **Source** | https://huggingface.co/guoyww/animatediff-motion-adapter-sdxl-beta |
| **Purpose** | Motion-enabled SDXL generation |
| **Strengths** | High resolution (1024x1024), good quality |
| **Weaknesses** | Requires SDXL base model |
| **VRAM** | 12GB+ |

---

#### B.1.4 I2VGen-XL (General)

| Property | Details |
|----------|---------|
| **Name** | I2VGen-XL |
| **Source** | https://huggingface.co/ali-vilab/i2vgen-xl |
| **Purpose** | Image-to-video generation |
| **Strengths** | Good image preservation, clear temporal structure |
| **Weaknesses** | Motion can be limited |
| **VRAM** | 10GB+ |

---

#### B.1.5 ZeroScope (General)

| Property | Details |
|----------|---------|
| **Name** | ZeroScope v2 |
| **Source** | https://huggingface.co/cerspense/zeroscope_v2_576w |
| **Purpose** | Fast, lightweight I2V |
| **Strengths** | Fast generation, reasonable quality |
| **Weaknesses** | Lower resolution |
| **VRAM** | 6GB |

---

### B.2 Depth Estimation

#### B.2.1 ZoeDepth

| Property | Details |
|----------|---------|
| **Name** | ZoeDepth |
| **Source** | https://huggingface.co/lllyasviel/ldm |
| **Purpose** | Metric depth estimation |
| **Strengths** | Best accuracy for most scenes, metric depth |
| **Weaknesses** | Slightly slower than MiDaS |
| **VRAM** | 2GB |

**Alternative**: Direct download from https://github.com/IDEA-Research/zoedepth

---

#### B.2.2 ZoeDepth-N

| Property | Details |
|----------|---------|
| **Name** | ZoeDepth-N |
| **Source** | https://huggingface.co/lllyasviel/ldm |
| **Purpose** | NYU-trained depth |
| **Strengths** | Best for indoor scenes |
| **Weaknesses** | Limited to indoor/general |
| **VRAM** | 2GB |

---

#### B.2.3 DPT (Dense Prediction Transformer)

| Property | Details |
|----------|---------|
| **Name** | DPT-Large |
| **Source** | https://huggingface.co/Intel/dpt-hybrid-midas |
| **Purpose** | High-accuracy depth |
| **Strengths** | Very accurate, good generalization |
| **Weaknesses** | Slower inference |
| **VRAM** | 3GB |

---

#### B.2.4 Marigold

| Property | Details |
|----------|---------|
| **Name** | Marigold |
| **Source** | https://huggingface.co/prs-eth/marigold-depth |
| **Purpose** | Diffusion-based depth |
| **Strengths** | Smooth, consistent depth maps |
| **Weaknesses** | Slower than traditional methods |
| **VRAM** | 6GB (FP16) |

---

### B.3 Segmentation

#### B.3.1 SAM (General)

| Property | Details |
|----------|---------|
| **Name** | SAM Vit-H |
| **Source** | https://huggingface.co/facebook/sam-vit-huge |
| **Purpose** | High-quality segmentation |
| **Strengths** | Best accuracy, any object |
| **Weaknesses** | Large model |
| **VRAM** | 2.5GB |

**SAM Variants**:
| Model | Parameters | VRAM |
|-------|------------|------|
| Vit-H | 632M | 2.5GB |
| Vit-L | 308M | 1GB |
| Vit-B | 91M | 400MB |

---

#### B.3.2 DeepLabV3+

| Property | Details |
|----------|---------|
| **Name** | DeepLabV3+ ResNet101 |
| **Source** | https://huggingface.co/google/deeplabv3_resnet101 |
| **Purpose** | Semantic segmentation |
| **Strengths** | Excellent for indoor/outdoor scenes |
| **Weaknesses** | Not instance-aware |
| **VRAM** | 1GB |

---

#### B.3.3 Mask2Former

| Property | Details |
|----------|---------|
| **Name** | Mask2Former |
| **Source** | https://huggingface.co/facebook/mask2former |
| **Purpose** | Universal segmentation |
| **Strengths** | Handles all segmentation types |
| **Weaknesses** | Large model |
| **VRAM** | 3GB |

---

#### B.3.4 MobileVOS

| Property | Details |
|----------|---------|
| **Name** | MobileVOS |
| **Source** | https://github.com/z-x-yang/MobileVOS |
| **Purpose** | Video object segmentation |
| **Strengths** | Fast, good for video |
| **Weaknesses** | Less accurate |
| **VRAM** | 500MB |

---

### B.4 Motion Conditioning

#### B.4.1 AnimateDiff Motion Adapter (General)

| Property | Details |
|----------|---------|
| **Name** | AnimateDiff Motion Adapter SDXL |
| **Source** | https://huggingface.co/guoyww/animatediff-motion-adapter-sdxl-beta |
| **Purpose** | Add motion to SDXL generations |
| **Strengths** | Smooth motion, multiple styles |
| **Weaknesses** | Requires SDXL base |
| **VRAM** | +2GB |

---

#### B.4.2 MM-DiT Video

| Property | Details |
|----------|---------|
| **Name** | MM-DiT |
| **Source** | Research (not yet public) |
| **Purpose** | Next-gen motion modeling |
| **Strengths** | Superior temporal consistency |
| **Weaknesses** | Not publicly available |

---

#### B.4.3 MotionClone

| Property | Details |
|----------|---------|
| **Name** | MotionClone |
| **Source** | https://github.com/MotionClone/MotionClone |
| **Purpose** | Clone motion from reference |
| **Strengths** | Precise motion transfer |
| **Weaknesses** | Requires motion reference |
| **VRAM** | 8GB |

---

### B.5 Frame Interpolation

#### B.5.1 RIFE (General)

| Property | Details |
|----------|---------|
| **Name** | RIFE v4 |
| **Source** | https://github.com/hzwer/RIFE |
| **Purpose** | Frame interpolation/upsampling |
| **Strengths** | Fast, excellent quality |
| **Weaknesses** | May have artifacts in complex motion |
| **VRAM** | 2GB |

---

#### B.5.2 CAIN (Channel Attention Is All You Need)

| Property | Details |
|----------|---------|
| **Name** | CAIN |
| **Source** | https://github.com/myungsub/CAIN |
| **Purpose** | Frame interpolation |
| **Strengths** | Good for repetitive motion |
| **Weaknesses** | Limited to 2x |
| **VRAM** | 1GB |

---

#### B.5.3 FLAVR

| Property | Details |
|----------|---------|
| **Name** | FLAVR |
| **Source** | https://github.com/tarun005/FLAVR |
| **Purpose** | Unidirectional frame interpolation |
| **Strengths** | Good for arbitrary timestep interpolation |
| **Weaknesses** | Uses more memory |
| **VRAM** | 3GB |

---

## Recommended Model Combinations

### Combination A1: Unrestricted (Desktop)

| Module | Model | VRAM |
|--------|-------|------|
| Depth | MiDaS v3.1 | 2GB |
| Segmentation | SAM Vit-B | 1GB |
| I2V | Open-SVD | 12GB |
| Interpolator | RIFE v4 | 2GB |
| **Total** | | **~17GB** |

---

### Combination A2: Unrestricted (Mid-Range Desktop)

| Module | Model | VRAM |
|--------|-------|------|
| Depth | MiDaS v3.1 | 2GB |
| Segmentation | MobileSAM | 500MB |
| I2V | ZeroScope | 6GB |
| Interpolator | RIFE v4 | 2GB |
| **Total** | | **~10.5GB** |

---

### Combination B1: General High-Quality (Desktop)

| Module | Model | VRAM |
|--------|-------|------|
| Depth | ZoeDepth | 2GB |
| Segmentation | SAM Vit-L | 1GB |
| I2V | SVD | 12GB |
| Interpolator | RIFE v4 | 2GB |
| **Total** | | **~17GB** |

---

### Combination B2: General (Laptop/Mid-Range)

| Module | Model | VRAM |
|--------|-------|------|
| Depth | DPT Hybrid | 1GB |
| Segmentation | SAM Vit-B | 1GB |
| I2V | AnimateDiff SD 1.5 | 6GB |
| Interpolator | RIFE v4 | 2GB |
| **Total** | | **~10GB** |

---

### Combination B3: General (Mobile-Ready)

| Module | Model | VRAM |
|--------|-------|------|
| Depth | MobileNet Depth | 200MB |
| Segmentation | MobileSAM | 300MB |
| I2V | ZeroScope (quantized) | 3GB |
| Interpolator | None (use generation FPS) | - |
| **Total** | | **~3.5GB** |

---

## Model Loading Architecture

### Model Registry

```python
MODEL_REGISTRY = {
    # I2V Generation
    "i2v": {
        "unrestricted": {
            "open-svd": {
                "repo": "camenduru/open-svd",
                "type": "svd",
                "vram": 12000,
            },
            "zeroscope": {
                "repo": "cerspense/zeroscope_v2_576w", 
                "type": "zeroscope",
                "vram": 6000,
            },
            "i2vgen-xl": {
                "repo": "ali-vilab/i2vgen-xl",
                "type": "i2vgen",
                "vram": 10000,
            },
        },
        "general": {
            "svd": {
                "repo": "stabilityai/stable-video-diffusion",
                "type": "svd",
                "vram": 12000,
            },
            "animatediff-sdxl": {
                "repo": "guoyww/animatediff-motion-adapter-sdxl-beta",
                "type": "animatediff",
                "vram": 10000,
            },
        },
    },
    
    # Depth Estimation
    "depth": {
        "unrestricted": {
            "midas": {
                "repo": "Intel/dpt-hybrid-midas",
                "type": "midas",
                "vram": 2000,
            },
            "depth-anything": {
                "repo": "LiheYoung/depth_anything_vitl14",
                "type": "depth-anything",
                "vram": 3000,
            },
        },
        "general": {
            "zoedepth": {
                "repo": "lllyasviel/ldm",
                "type": "zoedepth",
                "vram": 2000,
            },
            "marigold": {
                "repo": "prs-eth/marigold-depth",
                "type": "marigold",
                "vram": 6000,
            },
        },
    },
    
    # Segmentation
    "segmentation": {
        "unrestricted": {
            "sam-vit-huge": {
                "repo": "facebook/sam-vit-huge",
                "type": "sam",
                "vram": 2500,
            },
            "sam-vit-base": {
                "repo": "facebook/sam-vit-base", 
                "type": "sam",
                "vram": 400,
            },
            "mobile-sam": {
                "repo": "dh2811-group/mobile-sam",
                "type": "sam",
                "vram": 300,
            },
        },
        "general": {
            "sam-vit-large": {
                "repo": "facebook/sam-vit-large",
                "type": "sam",
                "vram": 1000,
            },
            "deeplabv3": {
                "repo": "google/deeplabv3_resnet101",
                "type": "deeplab",
                "vram": 1000,
            },
        },
    },
    
    # Frame Interpolation
    "interpolation": {
        "rife": {
            "repo": "hzwer/RIFE",
            "type": "rife",
            "vram": 2000,
        },
        "cain": {
            "repo": "myungsub/CAIN",
            "type": "cain",
            "vram": 1000,
        },
    },
}
```

---

## Mobile Model Recommendations

### Android (6GB RAM / 2GB VRAM)

| Module | Model | Notes |
|--------|-------|-------|
| Depth | MobileNet Depth | TFLite conversion needed |
| Segmentation | MobileSAM | Quantized to INT8 |
| I2V | ZeroScope | Reduced frames (8) |
| Interpolation | None | Use generated FPS |

### iOS (4GB RAM / 1.5GB VRAM)

| Module | Model | Notes |
|--------|-------|-------|
| Depth | MobileNet Depth | CoreML conversion needed |
| Segmentation | MobileSAM | CoreML conversion |
| I2V | ZeroScope | Quantized, 8 frames |
| Interpolation | None | Use generated FPS |

---

## Model Download & Caching

### HuggingFace Cache Structure

```
~/.cache/
├── huggingface/
│   └── hub/
│       ├── models--camenduru--open-svd/
│       ├── models--stabilityai--stable-video-diffusion/
│       ├── models--facebook--sam-vit-huge/
│       └── ...
└── torch/
    └── hub/
        └── ...
```

### Custom Cache Location

```python
import os
from pathlib import Path

# Set custom cache directory
os.environ["HF_HOME"] = str(Path.home() / ".cache" / "picture_aliver" / "models")
os.environ["TRANSFORMERS_CACHE"] = str(Path.home() / ".cache" / "picture_aliver" / "transformers")
```

---

## Summary Table

| Category | Best Desktop | Best Mobile | Best Unrestricted |
|----------|--------------|-------------|-------------------|
| I2V | SVD | ZeroScope | Open-SVD |
| Depth | ZoeDepth | MobileNet | MiDaS |
| Segmentation | SAM Vit-L | MobileSAM | SAM Vit-B |
| Interpolation | RIFE | None | RIFE |

---

*End of Model Selection Guide*