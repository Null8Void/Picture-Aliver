"""Type definitions for video generation."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch


@dataclass
class MotionGuidance:
    """Motion guidance for video generation.
    
    Attributes:
        flow_field: Optical flow field for motion transfer
        depth_map: Depth map for 3D motion
        segmentation: Segmentation masks for object-aware motion
        keypoints: Keypoint trajectories for character motion
        camera_motion: Camera motion parameters (pan, tilt, zoom)
    """
    flow_field: Optional[object] = None
    depth_map: Optional[object] = None
    segmentation: Optional[object] = None
    keypoints: Optional[torch.Tensor] = None
    camera_motion: Optional[Dict[str, float]] = None
    
    @property
    def has_motion(self) -> bool:
        return any([
            self.flow_field is not None,
            self.keypoints is not None,
            self.camera_motion is not None
        ])
    
    def get_motion_scale(self) -> float:
        """Get overall motion scale."""
        if self.camera_motion:
            return self.camera_motion.get("scale", 1.0)
        return 1.0


@dataclass
class GenerationConfig:
    """Configuration for video generation.
    
    Attributes:
        num_frames: Number of frames to generate
        fps: Frames per second
        resolution: Target resolution (H, W)
        guidance_scale: CFG guidance scale
        num_inference_steps: Number of denoising steps
        prompt: Text prompt for generation
        negative_prompt: Negative prompt
        seed: Random seed
        motion_strength: Strength of motion transfer (0-1)
        loop: Whether to make video loopable
        motion_mode: Motion mode ('flow', 'camera', 'keyframes', 'auto')
    """
    num_frames: int = 24
    fps: int = 8
    resolution: Tuple[int, int] = (512, 512)
    guidance_scale: float = 7.5
    num_inference_steps: int = 25
    prompt: str = ""
    negative_prompt: str = "blurry, low quality, artifacts, flickering, static"
    seed: Optional[int] = None
    motion_strength: float = 0.8
    loop: bool = False
    motion_mode: str = "auto"
    
    def __post_init__(self):
        if self.resolution is None:
            self.resolution = (512, 512)
        if isinstance(self.resolution, (list, tuple)) and len(self.resolution) == 2:
            pass
        else:
            self.resolution = (512, 512)


@dataclass
class VideoFrames:
    """Container for generated video frames.
    
    Attributes:
        frames: List of frames as tensors [C, H, W]
        timestamps: Timestamps for each frame
        metadata: Additional metadata
    """
    frames: List[torch.Tensor] = field(default_factory=list)
    timestamps: Optional[List[float]] = None
    metadata: Optional[Dict] = None
    
    def __len__(self) -> int:
        return len(self.frames)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.frames[idx]
    
    def append(self, frame: torch.Tensor, timestamp: Optional[float] = None) -> None:
        self.frames.append(frame)
        if self.timestamps is None:
            self.timestamps = []
        if timestamp is not None:
            self.timestamps.append(timestamp)
        elif self.timestamps is not None:
            self.timestamps.append(len(self.timestamps) / 8.0)
    
    @property
    def shape(self) -> Tuple[int, int, int, int]:
        """Get frame shape (C, H, W)."""
        if not self.frames:
            return (0, 0, 0, 0)
        frame = self.frames[0]
        if frame.dim() == 3:
            return tuple(frame.shape)
        return (frame.shape[0], frame.shape[1], frame.shape[2])
    
    @property
    def num_frames(self) -> int:
        return len(self.frames)
    
    def to_video(self) -> torch.Tensor:
        """Convert to video tensor [T, C, H, W]."""
        if not self.frames:
            return torch.zeros(0, 3, *self.resolution)
        return torch.stack(self.frames, dim=0)
    
    def to_list(self) -> List[np.ndarray]:
        """Convert to list of numpy arrays."""
        result = []
        for frame in self.frames:
            if isinstance(frame, torch.Tensor):
                frame = frame.cpu().numpy()
            if frame.shape[0] == 3:
                frame = frame.transpose(1, 2, 0)
            result.append(np.clip(frame * 255, 0, 255).astype(np.uint8))
        return result
    
    def to_pil(self) -> List:
        """Convert to list of PIL Images."""
        from PIL import Image
        
        result = []
        for frame in self.frames:
            if isinstance(frame, torch.Tensor):
                frame = frame.cpu().numpy()
            if frame.shape[0] == 3:
                frame = frame.transpose(1, 2, 0)
            frame = np.clip(frame, 0, 1)
            result.append(Image.fromarray((frame * 255).astype(np.uint8)))
        return result
    
    def apply_temporal_filter(
        self,
        window_size: int = 3,
        sigma: float = 1.0
    ) -> "VideoFrames":
        """Apply temporal smoothing to reduce flickering.
        
        Args:
            window_size: Size of temporal filter window
            sigma: Gaussian sigma for temporal weights
            
        Returns:
            Filtered VideoFrames
        """
        import scipy.ndimage as ndimage
        
        video_tensor = self.to_video()
        filtered = torch.zeros_like(video_tensor)
        
        for c in range(video_tensor.shape[1]):
            channel = video_tensor[:, c].cpu().numpy()
            for h in range(video_tensor.shape[2]):
                for w in range(video_tensor.shape[3]):
                    filtered[:, c, h, w] = torch.from_numpy(
                        ndimage.gaussian_filter1d(channel[:, h, w], sigma, mode='nearest')
                    )
        
        result = VideoFrames()
        for frame in filtered:
            result.append(frame)
        
        return result
    
    def upscale(self, scale: float = 2.0) -> "VideoFrames":
        """Upscale frames using interpolation.
        
        Args:
            scale: Scale factor
            
        Returns:
            Upscaled VideoFrames
        """
        result = VideoFrames()
        for frame in self.frames:
            h, w = frame.shape[-2:]
            new_h, new_w = int(h * scale), int(w * scale)
            upscaled = F.interpolate(
                frame.unsqueeze(0),
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False
            )
            result.append(upscaled.squeeze(0))
        return result
    
    @property
    def resolution(self) -> Tuple[int, int]:
        """Get frame resolution (H, W)."""
        if not self.frames:
            return (0, 0)
        frame = self.frames[0]
        return (frame.shape[-2], frame.shape[-1])
    
    def concatenate(self, other: "VideoFrames") -> "VideoFrames":
        """Concatenate with another VideoFrames."""
        result = VideoFrames()
        result.frames = self.frames + other.frames
        if self.timestamps and other.timestamps:
            result.timestamps = self.timestamps + [
                self.timestamps[-1] + t for t in other.timestamps
            ]
        return result


@dataclass
class SceneContext:
    """Scene context for video generation.
    
    Attributes:
        depth: Depth map
        segmentation: Segmentation masks
        semantics: Semantic labels
        objects: List of detected objects
        background: Background mask
        depth_order: Objects sorted by depth
    """
    depth: Optional[object] = None
    segmentation: Optional[object] = None
    semantics: Optional[Dict] = None
    objects: List[object] = field(default_factory=list)
    background: Optional[torch.Tensor] = None
    depth_order: Optional[List[int]] = None
    
    def get_layered_representation(self) -> Dict[int, torch.Tensor]:
        """Get objects as depth-ordered layers."""
        if self.depth_order is None:
            return {0: self.background} if self.background is not None else {}
        
        layers = {}
        for depth_idx, obj_idx in enumerate(self.depth_order):
            if obj_idx < len(self.objects):
                layers[depth_idx] = self.objects[obj_idx]
        
        return layers


import torch.nn.functional as F