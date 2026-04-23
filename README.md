# Image2Video AI

Production-ready AI system that converts a single image into a coherent animated video using modern techniques including diffusion models, motion priors, and scene understanding.

## Features

- **Depth Estimation**: ZoeDepth, MiDaS, Marigold support
- **Semantic Segmentation**: SAM (Segment Anything), DeepLabV3
- **Optical Flow**: RAFT, GMFlow, Farneback
- **Video Generation**: AnimateDiff, VideoLDM, custom diffusion
- **Temporal Consistency**: Gaussian smoothing, bilateral filtering, flicker reduction
- **GPU Accelerated**: CUDA/MPS support with mixed precision

## Installation

```bash
pip install -r requirements.txt
```

Or install in development mode:
```bash
pip install -e .
```

## Quick Start

### Python API

```python
from src import Image2Video

converter = Image2Video()
result = converter.convert("photo.jpg", "video.mp4", num_frames=24)
print(f"Generated {result.num_frames} frames in {result.processing_time:.2f}s")
```

### CLI

```bash
python -m src.bin.cli --input photo.jpg --output video.mp4 --num-frames 24 --fps 8
```

## Architecture

```
Input Image
    ├── Depth Estimation (ZoeDepth/MiDaS)
    ├── Semantic Segmentation (SAM)
    ├── Optical Flow Estimation (RAFT)
    │
    ▼
Scene Understanding
    ├── Depth Map
    ├── Object Masks
    ├── Motion Vectors
    │
    ▼
Video Generation
    ├── Diffusion Model
    ├── Motion Transfer
    ├── Temporal Consistency
    │
    ▼
Output Video
```

## Configuration

Edit `configs/default.yaml` or use the API:

```python
config = PipelineConfig(
    enable_depth=True,
    enable_segmentation=True,
    motion_mode="camera",  # camera, flow, zoom, pan
    num_frames=24,
    fps=8
)
```

## Command Line Options

```bash
python -m src.bin.cli \
    --input photo.jpg \
    --output video.mp4 \
    --prompt "gentle ocean waves" \
    --motion-mode camera \
    --num-frames 48 \
    --fps 12 \
    --guidance-scale 7.5
```

## Motion Modes

- **auto**: Automatically select based on content
- **camera**: Simulate camera movement (pan, tilt, zoom)
- **flow**: Use optical flow for motion transfer
- **keyframes**: Interpolate between keyframes
- **zoom**: Zoom in/out effect
- **pan**: Horizontal pan effect

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)

## License

Apache 2.0