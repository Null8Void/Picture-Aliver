"""Image2Video AI - Production Image to Video Synthesis System.

A comprehensive AI system that converts a single image into a coherent
animated video using state-of-the-art techniques including:
- Depth estimation (ZoeDepth, MiDaS, Marigold)
- Semantic segmentation (SAM, DeepLabV3)
- Optical flow estimation (RAFT, GMFlow, Farneback)
- Motion transfer with diffusion models (AnimateDiff, VideoLDM)
- Temporal consistency management

Example usage:
    >>> from src import Image2Video
    >>> converter = Image2Video()
    >>> result = converter.convert("photo.jpg", "video.mp4")

Or using the CLI:
    >>> python -m src.bin.cli --input photo.jpg --output video.mp4
"""

from .core.config import Config, load_config
from .core.device import DeviceManager, get_torch_device, get_optimal_device
from .core.pipeline import Image2VideoPipeline, PipelineConfig, PipelineResult
from .modules.depth import DepthEstimator, DepthMap
from .modules.segmentation import Segmentor, SegmentationMask
from .modules.motion import FlowEstimator, FlowField, MotionTrajectory
from .modules.generation import VideoGenerator, VideoFrames, GenerationConfig
from .utils.logger import setup_logger, get_logger

__version__ = "1.0.0"
__author__ = "AI Engineering Team"

__all__ = [
    "Image2Video",
    "Image2VideoPipeline",
    "PipelineConfig",
    "PipelineResult",
    "Config",
    "load_config",
    "DeviceManager",
    "get_torch_device",
    "get_optimal_device",
    "DepthEstimator",
    "DepthMap",
    "Segmentor",
    "SegmentationMask",
    "FlowEstimator",
    "FlowField",
    "MotionTrajectory",
    "VideoGenerator",
    "VideoFrames",
    "GenerationConfig",
    "setup_logger",
    "get_logger",
]


class Image2Video:
    """High-level API for image-to-video conversion."""
    
    def __init__(self, device: str = None, **kwargs):
        from .core.pipeline import Image2VideoPipeline, PipelineConfig
        from .core.device import get_torch_device
        
        self.device = get_torch_device(device)
        self.config = PipelineConfig(**kwargs)
        self.pipeline = Image2VideoPipeline(config=self.config, device=self.device)
        self._initialized = False
    
    def convert(self, image, output_path, **kwargs):
        if not self._initialized:
            self.pipeline.initialize()
            self._initialized = True
        return self.pipeline.process(image, output_path, **kwargs)
    
    def initialize(self):
        if not self._initialized:
            self.pipeline.initialize()
            self._initialized = True
    
    def clear_cache(self):
        self.pipeline.clear_cache()