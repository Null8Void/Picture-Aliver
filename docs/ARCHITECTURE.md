# Picture-Aliver: Image-to-Video System Architecture

## Document Info
- **Version**: 1.0
- **Date**: 2026-04-23
- **Status**: Architecture Design
- **Target Platforms**: Desktop (Windows/Linux/macOS), Android, iOS

---

## 1. Executive Summary

This document defines the complete system architecture for Picture-Aliver, a production-grade image-to-video synthesis system. The system converts a single input image into a coherent 2-10 second animated video using diffusion models, motion priors, and scene understanding.

**Design Philosophy**:
- Minimize hallucination through architectural constraints (not post-filtering)
- Enable model swapping without architecture changes
- Support offline operation with no external API dependencies
- Optimize for both desktop GPU and mobile deployment

---

## 2. System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PICTURE-ALIVER SYSTEM                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────┐    ┌───────────────┐   │
│  │  INPUT   │───▶│   SCENE      │───▶│   MOTION    │───▶│    VIDEO      │   │
│  │ IMAGE    │    │ UNDERSTANDING│    │ GENERATION │    │  SYNTHESIS    │   │
│  └──────────┘    └──────────────┘    └─────────────┘    └────────────��──┘   │
│       │                │                    │                   │           │
│       │                ▼                    ▼                   ▼           │
│       │          ┌────────────┐        ┌────────────┐      ┌─────────────┐   │
│       │          │  SCENE    │        │   MOTION   │      │ TEMPORAL    │   │
│       │          │ CONTEXT   │        │   PRIORS   │      │ STABILIZER  │   │
│       │          └────────────┘        └────────────┘      └─────────────┘   │
│       │                                                          │           │
│       ▼                                                          ▼           │
│  ┌────────────┐                                           ┌──────────────┐   │
│  │   MOBILE  │                                           │   EXPORT     │   │
│  │  LAYER    │                                           │   SYSTEM     │   │
│  └────────────┘                                           └──────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Module Specifications

### 3.1 Input Processing Module

**Role**: Prepare input image for downstream processing. Handle multiple input formats, normalize resolution, extract metadata, and detect image quality issues.

**Responsibilities**:
- Load images from file path, URL, PIL Image, numpy array, or tensor
- Detect and handle RGBA, grayscale, HDR formats
- Resolution normalization (target: 512x512 or 768x768 based on model)
- Aspect ratio preservation with intelligent cropping
- Image quality assessment (blur detection, compression artifact detection)
- EXIF metadata extraction for camera/sensor info
- Color space normalization (sRGB conversion)

**Inputs**:
```
{
  "source": Union[str, Path, PIL.Image, np.ndarray, torch.Tensor],
  "target_resolution": Tuple[int, int],  # (H, W)
  "preserve_aspect": bool,
  "metadata": Dict  # EXIF data if available
}
```

**Outputs**:
```
{
  "tensor": torch.Tensor,        # [C, H, W] normalized to [0, 1]
  "original_size": Tuple[int, int],
  "aspect_ratio": float,
  "color_profile": str,          # sRGB, Adobe RGB, etc.
  "quality_score": float,        # 0-1 quality assessment
  "dominant_colors": List,       # For palette preservation
  "quality_warnings": List[str]  # "blurry", "compressed", etc.
}
```

**GPU Acceleration**: None (CPU preprocessing only for I/O efficiency)

**Integration Notes**:
- Mobile: Handle camera roll access, HEIC/HEIF formats on iOS
- Android: Handle various camera app formats, rotation metadata
- Need EXIF rotation handling (images often rotated)

---

### 3.2 Scene Understanding Module

**Role**: Extract semantic understanding of the image to constrain video generation and reduce hallucination.

**Sub-modules**:

#### 3.2.1 Depth Estimator
- **Model Options** (General Use):
  - ZoeDepth (`ZoeD-M12`): Best accuracy for indoor/outdoor scenes
  - MiDaS v3 (`DPT-Large`): Balanced accuracy/speed
  - Marigold: Diffusion-based, smooth outputs
- **Model Options** (Unrestricted):
  - LeRes: Strong depth estimation from Landscape Models
- **Fallback**: Simple stereo-inspired depth if no GPU

**Inputs**: Preprocessed image tensor
**Outputs**: `DepthMap` with relative depth + confidence mask

#### 3.2.2 Semantic Segmentor
- **Model Options** (General Use):
  - SAM (Segment Anything Model): Vit-H for accuracy, Vit-B for speed
  - DeepLabV3+: Good for indoor scenes
  - SMOKE: Keypoint detection
- **Model Options** (Unrestricted):
  - OneFormer: Panoptic segmentation
- **Fallback**: Simple U-Net segmentation

**Inputs**: Preprocessed image tensor
**Outputs**: `SegmentationMask` with instance masks, class labels, confidence scores

#### 3.2.3 Scene Classifier
- **Model Options**:
  - CLIP: Zero-shot classification
  - SceneNet: Indoor scene classification
  - EfficientNet: General scene classification
- **Purpose**: Guide motion style selection and prompt generation

**Inputs**: Preprocessed image tensor
**Outputs**: Scene type labels, motion context hints

**GPU Acceleration**: Critical - all models run on GPU. Models loaded once at startup and kept in memory.

**Cache Strategy**:
```
GPU Memory Layout:
├── ZoeDepth Model: ~400MB
├── SAM Model: ~2.5GB (Vit-H)
├── CLIP Model: ~1.5GB
└── Available for generation: ~Remaining VRAM
```

**Integration Notes**:
- Mobile: May need model quantization (INT8) or smaller variants
- Consider running scene understanding on separate thread from generation
- Depth and segmentation can run in parallel

---

### 3.3 Motion Generation Module

**Role**: Generate motion priors based on scene understanding and motion style parameters.

**Sub-modules**:

#### 3.3.1 Motion Analyzer
**Purpose**: Analyze static image for inherent motion cues
- Flow patterns in textures
- Depth gradients (near/far parallax)
- Object placement and size
- Horizon line position

#### 3.3.2 Motion Scheduler
**Purpose**: Create smooth motion trajectory over video duration

**Motion Style Presets**:
| Style | Description | Parameters |
|-------|-------------|------------|
| `cinematic` | Slow dolly/pan, subtle zoom | Pan: 0.02 rad, Zoom: 0.05x, Duration: 3-5s |
| `subtle` | Minimal movement, breathing effect | Pan: 0.01 rad, Zoom: 0.02x, Duration: 2-3s |
| `environmental` | Wind, water, leaves | Wave amplitude: 0.05, Frequency: 0.5Hz |
| `orbital` | Circular camera path | Radius: 0.1, Speed: 0.3 rad/s |
| `zoom-in` | Slow zoom in | Scale: 1.0 → 1.2 |
| `zoom-out` | Slow zoom out | Scale: 1.2 → 1.0 |
| `pan-left` | Horizontal pan | Velocity: 0.05 px/frame |
| `pan-right` | Horizontal pan | Velocity: -0.05 px/frame |

#### 3.3.3 Flow Field Generator
**Purpose**: Generate dense optical flow fields for motion transfer

**Model Options**:
- RAFT: Highest accuracy, slower
- GMFlow: Transformer-based, good accuracy
- Farneback: CPU fallback, fast but lower quality

**Outputs**:
```
{
  "flow_fields": List[torch.Tensor],  # [T, 2, H, W] per frame
  "keyframe_indices": List[int],
  "trajectory": MotionTrajectory,
  "camera_motion": Dict{
    "pan": float,
    "tilt": float,
    "zoom": float,
    "roll": float
  }
}
```

**GPU Acceleration**: Flow estimation runs on GPU. Key for smooth motion.

**Integration Notes**:
- Pre-compute flow fields for all frames upfront
- Store in GPU memory during generation
- Mobile: May need reduced resolution flow fields

---

### 3.4 Video Synthesis Module

**Role**: Generate video frames using latent diffusion with motion conditioning.

**Architecture**:
```
┌─────────────────────────────────────────────────────────┐
│                 LATENT DIFFUSION PIPELINE               │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────┐    ┌──────────┐    ┌────────────┐           │
│  │ NOISE   │───▶│ UNET     │◀───│ MOTION     │           │
│  │ LATENT  │    │ 3D       │    │ EMBEDDING  │           │
│  └─────────┘    └──────────┘    └────────────┘           │
│       │              │                    │               │
│       │              ▼                    │               │
│       │    ┌────────────────┐             │               │
│       │    │   TEMPORAL     │             │               │
│       │    │   ATTENTION    │─────────────┘               │
│       │    └────────────────┘                             │
│       │              │                                    │
│       │              ▼                                    │
│       │    ┌────────────────┐                            │
│       │    │  VAE DECODER   │                            │
│       │    └────────────────┘                            │
│       │              │                                    │
│       ▼              ▼                                    │
│  ┌─────────────────────────────────┐                     │
│  │      FRAME BATCH [T, C, H, W]   │                     │
│  └─────────────────────────────────┘                     │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**Model Options** (General Use):
| Model | Resolution | Frames | Speed | Quality |
|-------|------------|--------|-------|---------|
| AnimateDiff SDXL | 1024 | 16-24 | Medium | Excellent |
| AnimateDiff SD 1.5 | 512 | 16-32 | Fast | Good |
| SVD (Stable Video Diffusion) | 576 | 14-25 | Medium | Excellent |
| I2VGen-XL | 512 | 16 | Medium | Good |
| ZeroScroll | 512 | 8-16 | Fast | Good |

**Model Options** (Unrestricted):
- OpenGIF: Higher quality, less content restrictions
- ModelScope: Large model with strong motion
- LaVie: High quality, longer sequences

**Motion Adapter Options**:
- `animatediff-motion-adapter-sdxl-beta`: Best for SDXL
- `animatediff-motion-adapter`: For SD 1.5
- `motion-module`: For SVD

**Conditioning Inputs**:
1. **Image Embedding**: VAE encode input image, concatenate to latent
2. **Motion Embedding**: Camera motion vectors, flow fields
3. **Depth Conditioning**: Depth map for 3D consistency
4. **Segmentation Mask**: Object boundaries preserved

**GPU Requirements**:
```
Minimum VRAM: 8GB (SD 1.5 + AnimateDiff)
Recommended: 12GB+ (SDXL + AnimateDiff)
```

**Batch Processing**:
- Generate frames in chunks of 8-16
- Apply CFG with motion guidance
- Use EMA for stable denoising

**Integration Notes**:
- Mobile: Consider quantized models (GPTQ, AWQ)
- May need frame-by-frame generation for <6GB VRAM
- Model downloaded on first use (Hugging Face cache)

---

### 3.5 Temporal Stabilization Module

**Role**: Ensure temporal consistency across generated frames, eliminate flickering and warping.

**Architectural Approach**:
> **Philosophy**: Stabilization through architecture, NOT post-filtering.

Primary stabilization is done during generation via:
1. Temporal attention in UNet
2. Motion-consistent conditioning
3. Scene-aware denoising schedules

**Secondary stabilization** (applied after generation):

#### 3.5.1 Temporal Smoothing
**Methods**:
- Gaussian smoothing along time axis
- Bilateral filtering (preserve edges)
- Median filtering (remove outliers)

**Parameters**:
```
{
  "method": "gaussian",
  "sigma_spatial": 0.5,  # Spatial blur strength
  "sigma_temporal": 1.0, # Temporal blur strength
  "window_size": 3        # Frames to consider
}
```

#### 3.5.2 Flicker Reduction
**Detection**:
- Inter-frame luminance variance
- Color channel deviation
- Edge consistency measure

**Correction**:
- Histogram matching between adjacent frames
- Gain adjustment for brightness normalization
- Color temperature smoothing

#### 3.5.3 Loop Consistency (Optional)
For seamless loops:
- Blend last N frames with first N frames
- Match motion trajectory endpoints
- Smooth transition zone

**GPU Acceleration**: All operations tensor-based, fully GPU-accelerated.

**Integration Notes**:
- Can be disabled if latency is critical
- Mobile: Lighter smoothing (sigma_temporal: 0.5)

---

### 3.6 Post-Processing Module

**Role**: Enhance visual quality and apply final refinements.

**Sub-modules**:

#### 3.6.1 Color Grading
- Match original image color characteristics
- Smooth tone transitions
- Saturation normalization

#### 3.6.2 Sharpening
- Intelligent sharpening avoiding noise amplification
- Edge-aware sharpening strength
- Contrast enhancement

#### 3.6.3 Artifact Reduction
- Compression artifact smoothing
- Ghosting reduction near motion boundaries
- Color banding elimination

#### 3.6.4 Resolution Upscaling (Optional)
**Model Options**:
- Real-ESRGAN: Best quality
- Real-CUGAN: Fast
- SwinIR: Balanced
- Waifu2x: Anime optimized

**Integration Notes**:
- Mobile: Skip upscaling by default (battery saving)
- Desktop: Optional 2x upscaling

---

### 3.7 Export System Module

**Role**: Encode and save video in various formats with proper metadata.

**Supported Formats**:
| Format | Codec | Pros | Cons |
|--------|-------|------|------|
| MP4 | H.264 | Universal support | Quality loss |
| MP4 | H.265 | Better compression | Limited support |
| WebM | VP9 | Web optimized | Less compatible |
| MOV | ProRes | High quality | Large files |
| GIF | N/A | No player needed | Limited colors |

**Encoding Options**:
```
{
  "format": "mp4",
  "codec": "libx264",
  "crf": 23,           # 18-28, lower is better
  "preset": "medium",   # ultrafast, fast, medium, slow
  "fps": 8,
  "audio": None,        # No audio by default
  "metadata": {
    "title": str,
    "description": str,
    "fps": int,
    "duration": float
  }
}
```

**GPU Acceleration**:
- Use NVENC (NVIDIA) or AMF (AMD) for hardware encoding
- Fallback to CPU encoding (ffmpeg) if no GPU encoder

**Integration Notes**:
- Mobile: Use platform-specific codecs (MediaCodec on Android)
- iOS: AVAssetWriter with hardware encoding
- Consider WebM for web sharing

---

### 3.8 Mobile Integration Layer

**Role**: Abstract platform-specific implementations for Android/iOS/Desktop.

**Architecture**:
```
┌─────────────────────────────────────────────────────────────────┐
│                      PYTHON CORE (Portable)                     │
├─────────────────────────────────────────────────────────────────┤
│  ImageProcessor │ SceneUnderstanding │ VideoSynthesizer │ Export │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BRIDGE LAYER                                  │
├───────────────────────────┬─────────────────────────────────────┤
│      Android Bridge        │         iOS Bridge                  │
│  ┌───────────────────┐    │    ┌────────────��──────┐           │
│  │  Chaquopy (Python)│    │    │   PythonKit       │           │
│  │  or JNI bindings  │    │    │   or PyMobile     │           │
│  └───────────────────┘    │    └───────────────────┘           │
└───────────────────────────┴─────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    NATIVE LAYER                                  │
├───────────────────────────┬─────────────────────────────────────┤
│     Android (Kotlin)      │           iOS (Swift)                │
│  ┌───────────────────┐    │    ┌───────────────────┐           │
│  │ Activity/Fragment │    │    │  UIViewController │           │
│  │ Camera/Gallery    │    │    │  PhotoKit/AVFoun  │           │
│  │ MediaCodec        │    │    │  AVAssetWriter    │           │
│  └───────────────────┘    │    └───────────────────┘           │
└───────────────────────────┴─────────────────────────────────────┘
```

**Android Integration**:
- **Python Runtime**: Chaquopy or BeeWare
- **Model Format**: TorchScript (.pt) for inference
- **Quantization**: INT8 or FP16 for mobile GPUs
- **Memory Management**: Stream frames, don't cache all

**iOS Integration**:
- **Python Runtime**: PythonKit or custom build
- **Model Format**: CoreML conversion or TorchScript
- **Quantization**: ML Compute with bfloat16
- **Memory**: Same as Android, avoid frame caching

**Shared Mobile Considerations**:
- Model download manager (WiFi-only option)
- Progress callbacks for long generation
- Cancellation support
- Background processing with notifications
- Battery optimization (reduce quality if low battery)

---

## 4. Pipeline Flow

### 4.1 Full Processing Pipeline

```
INPUT IMAGE
    │
    ▼
┌──────────────────────────────────────────────────────────────────────┐
│ STEP 1: INPUT PROCESSING (CPU)                                        │
│ • Load and validate image                                            │
│ • Resolution normalization                                            │
│ • Color correction                                                    │
│ • Quality assessment                                                  │
└──────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────────┐
│ STEP 2: SCENE UNDERSTANDING (GPU - Parallel)                          │
│                                                                      │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│   │   DEPTH     │  │ SEGMENTATION│  │   SCENE     │                 │
│   │ ESTIMATION  │  │             │  │ CLASSIFIER  │                 │
│   │             │  │             │  │             │                 │
│   │ ZoeDepth    │  │ SAM         │  │ CLIP        │                 │
│   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                 │
│          │                │                │                         │
│          └────────────────┼────────────────┘                         │
│                           ▼                                          │
│                    SCENE CONTEXT                                      │
│          (Depth, Masks, Classes, Motion Hints)                        │
└──────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────────┐
│ STEP 3: MOTION GENERATION (GPU)                                       │
│                                                                      │
│   • Analyze scene for motion cues                                     │
│   • Generate camera trajectory (pan/zoom)                            │
│   • Create flow fields for motion transfer                           │
│   • Schedule keyframes                                               │
│                                                                      │
│   OUTPUT: MotionGuidance(scene_context, style_params)                │
└──────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────────┐
│ STEP 4: VIDEO SYNTHESIS (GPU - Main Computation)                     │
│                                                                      │
│   FOR each frame batch (8-16 frames):                                 │
│   ┌─────────────────────────────────────────────────────────────┐    │
│   │ • Add noise to latent                                        │    │
│   │ • Encode conditioning (image, depth, motion)                 │    │
│   │ • UNet denoising with temporal attention                     │    │
│   │ • Decode to pixel space                                      │    │
│   │ • Apply temporal consistency loss                            │    │
│   └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│   OUTPUT: VideoFrames[batch]                                          │
└──────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────────┐
│ STEP 5: TEMPORAL STABILIZATION (GPU)                                  │
│                                                                      │
│   • Gaussian temporal smoothing                                       │
│   • Flicker detection and correction                                 │
│   • Histogram matching between frames                                 │
│   • Loop blending (if enabled)                                        │
│                                                                      │
│   OUTPUT: Stabilized VideoFrames                                      │
└──────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────────┐
│ STEP 6: POST-PROCESSING (GPU)                                         │
│                                                                      │
│   • Color grading                                                     │
│   • Sharpening                                                        │
│   • Artifact reduction                                                │
│   • Resolution upscaling (optional)                                   │
│                                                                      │
│   OUTPUT: Enhanced VideoFrames                                         │
└──────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────────┐
│ STEP 7: EXPORT (CPU/GPU)                                              │
│                                                                      │
│   • Encode to target format (MP4/WebM/GIF)                            │
│   • Hardware encoding if available                                    │
│   • Add metadata                                                      │
│   • Save to output path                                               │
│                                                                      │
│   OUTPUT: Video file saved to disk                                    │
└──────────────────────────────────────────────────────────────────────┘
    │
    ▼
FINAL OUTPUT
```

### 4.2 Streaming Pipeline (Low Memory)

For devices with < 6GB VRAM:

```
┌──────────────────────────────────────────────────────────────────────┐
│ STREAMING MODE (Frame-by-Frame Generation)                            │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│ INIT: Load all models, preload VAE, encode input image as condition   │
│                                                                      │
│ FOR each target frame t = 0 to num_frames:                            │
│   │                                                                    │
│   ├─▶ LOAD motion state for frame t                                   │
│   │       └── Camera position, flow field, keyframe data               │
│   │                                                                    │
│   ├─▶ GENERATE frame t (diffusion pass)                               │
│   │       └── Conditioned on: input image, previous frame, motion     │
│   │                                                                    │
│   ├─▶ STABILIZE frame t (lightweight)                                 │
│   │       └── Simple temporal blend with previous                     │
│   │                                                                    │
│   ├─▶ ENCODE frame t to output                                        │
│   │       └── Write directly to video stream (no frame caching)       │
│   │                                                                    │
│   └─▶ UPDATE state for next frame                                     │
│           └── Store reference frame, motion state                      │
│                                                                      │
│ COMPLETE: Finalize video file                                         │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 5. GPU Memory Management

### 5.1 Memory Layout (Desktop - 12GB VRAM)

```
┌────────────────────────────────────────────────────────────────────┐
│ GPU MEMORY (12GB)                                                   │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ ┌──────────────────────────────────────┐  ~500MB                    │
│ │ Depth Estimator (ZoeDepth)           │                            │
│ └──────────────────────────────────────┘                            │
│                                                                     │
│ ┌──────────────────────────────────────┐  ~2.5GB                    │
│ │ SAM Model (Vit-H)                    │                            │
│ └──────────────────────────────────────┘                            │
│                                                                     │
│ ┌──────────────────────────────────────┐  ~1.5GB                    │
│ │ CLIP Encoder                        │                            │
│ └──────────────────────────────────────┘                            │
│                                                                     │
│ ┌──────────────────────────────────────┐  ~3GB                      │
│ │ Stable Diffusion + Motion Adapter    │                            │
│ └──────────────────────────────────────┘                            │
│                                                                     │
│ ┌──────────────────────────────────────┐  ~2GB                      │
│ │ VAE Decoder (frame buffer)           │                            │
│ └──────────────────────────────��───────┘                            │
│                                                                     │
│ ┌──────────────────────────────────────┐  ~500MB                    │
│ │ Flow Fields & Motion Data            │                            │
│ └──────────────────────────────────────┘                            │
│                                                                     │
│ ┌──────────────────────────────────────┐  ~2GB (Available)          │
│ │ Available for processing             │                            │
│ └──────────────────────────────────────┘                            │
│                                                                     │
└─────────────────────────────────────────────���──────────────────────┘
```

### 5.2 Mobile Memory Strategy (< 2GB VRAM)

```
MOBILE STRATEGY (ON-DEMAND LOADING):
├── At Startup:
│   └── Load only Scene Understanding models
│       └── SAM (Vit-B, ~500MB) or skip if memory critical
│
├── During Scene Understanding:
│   └── Load depth estimator temporarily, unload after
│   └── Load segmentation, run, then optionally unload
│
├── During Generation:
│   └── Keep only SD model in memory
│   └── Process frames in batches of 4
│   └── No SAM/depth in memory during generation
│
└── After Export:
    └── Free all model memory
```

---

## 6. Model Loading & Reuse Strategy

### 6.1 Singleton Model Manager

```python
class ModelManager:
    """Centralized model loading with caching."""

    _instance = None
    _models = {}

    @classmethod
    def get_model(cls, model_type: str, config: dict) -> nn.Module:
        """
        Get or load model. Models are loaded once and cached.

        Caching Key: (model_type, model_variant, precision)
        Example: ("depth", "zoedepth_kitti", "fp16")
        """
        cache_key = (model_type, config.get("variant"), config.get("precision"))

        if cache_key not in cls._models:
            cls._models[cache_key] = cls._load_model(model_type, config)

        return cls._models[cache_key]

    @classmethod
    def unload_model(cls, model_type: str):
        """Free model memory when no longer needed."""
        # Find and remove all keys matching model_type
        keys_to_remove = [k for k in cls._models.keys() if k[0] == model_type]
        for key in keys_to_remove:
            del cls._models[key]
```

### 6.2 Load Order

```
1. At Application Startup:
   ├── Load CLIP (always needed for scene understanding)
   └── Load Depth Estimator (needed early for scene context)

2. After Scene Understanding:
   ├── Load Segmentation Model (can be unloaded after use)
   └── Unload Depth Estimator (if memory needed)

3. Before Video Synthesis:
   ├── Load Stable Diffusion + Motion Adapter
   ├── Load VAE (kept for decoding)
   └── Unload Segmentation if still in memory

4. During Generation:
   └── All models stay loaded (generation requires all)

5. After Export Complete:
   └── Free all models (optional, keep warm if repeated use)
```

---

## 7. Data Flow Between Modules

### 7.1 Primary Data Structures

```python
@dataclass
class SceneContext:
    """Shared context passed between modules."""
    depth_map: Optional[torch.Tensor]  # [1, H, W]
    depth_confidence: torch.Tensor     # [1, H, W]
    segmentation: List[SegmentationMask]
    semantic_classes: Dict[str, float]
    scene_type: str
    dominant_colors: List[Color]
    horizon_position: float  # 0-1, 0.5 = center
    focal_point: Point        # Computed center of interest


@dataclass
class MotionGuidance:
    """Motion parameters for generation."""
    style: MotionStyle  # Enum: CINEMATIC, SUBTLE, ENVIRONMENTAL, etc.
    camera_trajectory: List[CameraPose]  # Per-frame camera state
    flow_fields: List[torch.Tensor]      # [T, 2, H, W] optical flow
    keyframe_indices: List[int]
    strength: float  # 0-1 motion intensity


@dataclass
class VideoFrames:
    """Generated video frames."""
    frames: List[torch.Tensor]  # [T, C, H, W] in range [0, 1]
    fps: int
    resolution: Tuple[int, int]
    timestamp: float  # Generation timestamp
    metadata: Dict     # Any additional metadata
```

### 7.2 Module Interface Summary

```
Module              Input                    Output
────────────────────────────────────────────────────────────────
InputProcessor      raw_image               PreprocessedImage
                   
SceneUnderstanding  PreprocessedImage ───▶ SceneContext
                   
MotionGeneration    SceneContext +          MotionGuidance
                    style_params ─────────▶
                   
VideoSynthesizer    PreprocessedImage +      VideoFrames
                    SceneContext + ───────▶
                    MotionGuidance
                   
TemporalStabilizer  VideoFrames ──────────▶ VideoFrames
                   
PostProcessor       VideoFrames ──────────▶ EnhancedVideoFrames
                   
ExportSystem        EnhancedVideoFrames ──▶ VideoFile
```

---

## 8. Integration Concerns & Mitigation

### 8.1 PC Integration Concerns

| Concern | Mitigation |
|---------|------------|
| Varying GPU capabilities | Detect GPU at startup, adjust quality presets |
| Multiple GPUs | Allow GPU selection, default to fastest |
| Driver compatibility | Test CUDA/ROCm versions, fallback gracefully |
| User storage space | Check available space before model download |
| Memory pressure | Monitor system memory, unload models when needed |
| DLL dependencies | Bundle all dependencies, use TorchScript where possible |

### 8.2 Android Integration Concerns

| Concern | Mitigation |
|---------|------------|
| Thermal throttling | Monitor device temperature, reduce quality if hot |
| Memory fragmentation | Use object pooling, minimize allocations |
| Battery drain | Offer "Eco mode" with reduced quality |
| App size (models are large) | Offer optional model downloads, use ONNX quantization |
| Permission handling | Graceful degradation if camera/storage denied |
| Background processing | Use WorkManager, show notification during generation |
| ANR (App Not Responding) | Background thread for all generation, progress callbacks |
| Native library compatibility | NDK compatibility matrix, prebuilt APKs per ABI |

**Android Specific Architecture**:
```
Android App (Kotlin)
    │
    ├── Python Runtime (Chaquopy)
    │   │
    │   └── PictureAliverCore (Python)
    │       ├── InputProcessor
    │       ├── SceneUnderstanding (with model manager)
    │       ├── MotionGeneration
    │       ├── VideoSynthesizer
    │       └── ExportSystem
    │
    └── Native Bridge
        ├── Model download service
        ├── Camera/Gallery integration
        ├── Hardware encoding (MediaCodec)
        └── Notification service
```

### 8.3 iOS Integration Concerns

| Concern | Mitigation |
|---------|------------|
| Swift-Python interop | Use PythonKit or embedded Python interpreter |
| Memory limits | Aggressive model unloading, streaming generation |
| CoreML optimization | Convert models to CoreML for GPU acceleration |
| App Store size limits | On-demand model downloads |
| Privacy | All processing on-device, no data leaves phone |
| Background processing | Limited, use AVAudioSession tricks or background tasks |

**iOS Specific Architecture**:
```
iOS App (Swift)
    │
    ├── PythonKit or Embedded Python
    │   │
    │   └── PictureAliverCore
    │       ├── [Same modules as Android]
    │       └── iOS-specific file paths
    │
    └── Native Layer
        ├── PhotoKit for gallery access
        ├── AVFoundation for camera
        ├── AVAssetWriter for encoding
        └── Background task handling
```

---

## 9. Error Handling & Recovery

### 9.1 Module-Level Error Handling

```
Each module implements:
├── validate_input(): Pre-flight checks
├── process(): Main processing with error catching
├── recover(): Graceful degradation on failure
└── cleanup(): Resource release on error
```

### 9.2 Fallback Chains

```
Depth Estimation:
  ZoeDepth → MiDaS → Simple depth (no GPU fallback)

Segmentation:
  SAM Vit-H → SAM Vit-B → DeepLabV3 → No segmentation

Video Generation:
  SVD → AnimateDiff SDXL → AnimateDiff SD 1.5 → Simple motion

Encoding:
  NVENC/AMF → ffmpeg CPU → Pillow GIF (last resort)
```

---

## 10. Performance Targets

| Metric | Desktop | Android | iOS |
|--------|---------|---------|-----|
| Generation speed (frames/sec) | 2-4 | 0.5-1 | 0.5-1 |
| Total time (24 frames, 8fps) | 6-12s | 24-48s | 24-48s |
| Peak memory (VRAM) | 10-12GB | 2-3GB | 2-3GB |
| Peak memory (RAM) | 4-6GB | 1-2GB | 1-2GB |
| Output file size (24 frames) | 2-5MB | 2-5MB | 2-5MB |

---

## 11. Summary of Integration-Ready Design

### Key Architectural Decisions for Easy Integration

1. **Python Core is Portable**: All modules are pure Python, no OS-specific code
2. **Model Manager is Singleton**: Enables easy lifecycle management on mobile
3. **Clear Module Boundaries**: Each module has defined inputs/outputs
4. **Streaming Support**: Architecture supports frame-by-frame processing
5. **Memory Management**: Explicit model loading/unloading for mobile constraints
6. **Error Recovery**: Fallback chains prevent complete failure
7. **Quantization Ready**: All models support INT8/FP16 conversion paths
8. **Export is Pluggable**: New codecs/formats can be added without core changes

### Areas Requiring Platform-Specific Work

1. **Python Runtime**: Chaquopy for Android, PythonKit for iOS
2. **Model Format**: TorchScript for Android, CoreML for iOS
3. **File I/O**: Platform-specific paths and permissions
4. **Video Encoding**: MediaCodec for Android, AVAssetWriter for iOS
5. **Progress UI**: Platform-specific notification systems
6. **Power Management**: Android WorkManager, iOS Background Tasks

---

## 12. Next Steps

1. Review this architecture document
2. Confirm model choices match use case (general vs unrestricted)
3. Define specific quality presets (low/medium/high)
4. Design mobile UI/UX flow
5. Begin implementation with core pipeline

---

*End of Architecture Document*