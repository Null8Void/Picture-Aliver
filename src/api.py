"""Python API for Image2Video AI system."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch

from .core.pipeline import Image2VideoPipeline, PipelineConfig, PipelineResult
from .core.config import Config
from .core.device import get_torch_device, DeviceManager


class Image2Video:
    """High-level API for image-to-video conversion.
    
    Example usage:
        >>> from image2video import Image2Video
        >>> converter = Image2Video()
        >>> result = converter.convert("photo.jpg", "output.mp4")
        >>> print(f"Generated {result.num_frames} frames")
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        config: Optional[dict] = None
    ):
        """Initialize Image2Video converter.
        
        Args:
            device: Compute device ('cuda', 'cpu', 'mps')
            config: Optional configuration dictionary
        """
        self.device = get_torch_device(device) if device else None
        
        pipeline_config = PipelineConfig(
            enable_depth=True,
            enable_segmentation=True,
            enable_motion=True,
            enable_consistency=True,
            use_3d_effects=True,
            motion_mode="auto",
            verbose=False
        )
        
        if config:
            for key, value in config.items():
                if hasattr(pipeline_config, key):
                    setattr(pipeline_config, key, value)
        
        self.pipeline = Image2VideoPipeline(
            config=pipeline_config,
            device=self.device
        )
        
        self._initialized = False
    
    def convert(
        self,
        image: Union[str, Path],
        output_path: Union[str, Path],
        **kwargs
    ) -> PipelineResult:
        """Convert an image to video.
        
        Args:
            image: Input image path
            output_path: Output video path
            **kwargs: Additional generation parameters
            
        Returns:
            PipelineResult with generated video
        """
        if not self._initialized:
            self.pipeline.initialize()
            self._initialized = True
        
        return self.pipeline.process(
            image=str(image),
            output_path=str(output_path),
            **kwargs
        )
    
    def convert_simple(
        self,
        image: Union[str, Path],
        output_path: Union[str, Path],
        num_frames: int = 24,
        motion_mode: str = "camera",
        fps: int = 8
    ) -> PipelineResult:
        """Simple conversion with minimal parameters.
        
        Args:
            image: Input image path
            output_path: Output video path
            num_frames: Number of frames
            motion_mode: Motion mode
            fps: Frames per second
            
        Returns:
            PipelineResult
        """
        return self.convert(
            image=image,
            output_path=output_path,
            num_frames=num_frames,
            motion_mode=motion_mode,
            fps=fps
        )
    
    def initialize(self) -> None:
        """Initialize the pipeline components."""
        if not self._initialized:
            self.pipeline.initialize()
            self._initialized = True
    
    def clear_cache(self) -> None:
        """Clear GPU memory cache."""
        self.pipeline.clear_cache()
    
    def get_device_info(self) -> dict:
        """Get information about the current device."""
        dm = DeviceManager()
        return {
            "name": dm.current_device.name,
            "type": dm.current_device.type,
            "memory_gb": dm.current_device.memory_gb
        }


class Image2VideoConfig:
    """Configuration helper for Image2Video.
    
    Example:
        >>> config = Image2VideoConfig()
        >>> config.set_resolution(512)
        >>> config.set_motion_mode("camera")
        >>> config.disable_depth()
        >>> converter = Image2Video(config=config.to_dict())
    """
    
    def __init__(self):
        self._config = {
            "enable_depth": True,
            "enable_segmentation": True,
            "enable_motion": True,
            "enable_consistency": True,
            "use_3d_effects": True,
            "motion_mode": "auto",
            "verbose": False
        }
    
    def set_resolution(self, resolution: int) -> "Image2VideoConfig":
        """Set output resolution."""
        self._config["resolution"] = resolution
        return self
    
    def set_motion_mode(self, mode: str) -> "Image2VideoConfig":
        """Set motion mode."""
        valid_modes = ["auto", "camera", "flow", "keyframes", "zoom", "pan"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid motion mode: {mode}")
        self._config["motion_mode"] = mode
        return self
    
    def enable_depth(self) -> "Image2VideoConfig":
        """Enable depth estimation."""
        self._config["enable_depth"] = True
        return self
    
    def disable_depth(self) -> "Image2VideoConfig":
        """Disable depth estimation."""
        self._config["enable_depth"] = False
        return self
    
    def enable_segmentation(self) -> "Image2VideoConfig":
        """Enable segmentation."""
        self._config["enable_segmentation"] = True
        return self
    
    def disable_segmentation(self) -> "Image2VideoConfig":
        """Disable segmentation."""
        self._config["enable_segmentation"] = False
        return self
    
    def enable_consistency(self) -> "Image2VideoConfig":
        """Enable temporal consistency."""
        self._config["enable_consistency"] = True
        return self
    
    def disable_consistency(self) -> "Image2VideoConfig":
        """Disable temporal consistency."""
        self._config["enable_consistency"] = False
        return self
    
    def set_num_frames(self, num_frames: int) -> "Image2VideoConfig":
        """Set number of frames."""
        self._config["num_frames"] = num_frames
        return self
    
    def set_fps(self, fps: int) -> "Image2VideoConfig":
        """Set frames per second."""
        self._config["fps"] = fps
        return self
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return self._config.copy()


def convert_image(
    image: Union[str, Path],
    output_path: Union[str, Path],
    **kwargs
) -> PipelineResult:
    """Convenience function to convert a single image.
    
    Example:
        >>> result = convert_image("photo.jpg", "video.mp4", num_frames=24)
    """
    converter = Image2Video()
    return converter.convert(image, output_path, **kwargs)


__all__ = [
    "Image2Video",
    "Image2VideoConfig",
    "convert_image",
    "Image2VideoPipeline",
    "PipelineConfig",
    "PipelineResult",
]


import torch.nn.functional as F
from .modules.generation import VideoFrames, GenerationConfig, MotionGuidance
from .modules.depth import DepthEstimator, DepthMap
from .modules.segmentation import Segmentor, SegmentationMask
from .modules.motion import FlowEstimator, FlowField