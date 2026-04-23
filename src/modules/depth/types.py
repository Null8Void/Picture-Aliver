"""Type definitions for depth estimation."""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch


@dataclass
class DepthMap:
    """Container for depth estimation output.
    
    Attributes:
        depth: Raw depth values as tensor [H, W] or numpy array
        normalized: Depth values normalized to [0, 1] range
        inverse: Inverse depth (1/depth) for metric consistency
        scale: Absolute scale factor if available
        intrinsics: Camera intrinsics if available
    """
    depth: torch.Tensor
    normalized: Optional[torch.Tensor] = None
    inverse: Optional[torch.Tensor] = None
    scale: Optional[float] = None
    intrinsics: Optional[dict] = None
    
    def __post_init__(self):
        if self.normalized is None:
            self._normalize()
        if self.inverse is None:
            self._compute_inverse()
    
    def _normalize(self) -> None:
        """Normalize depth to [0, 1] range."""
        d_min = self.depth.min()
        d_max = self.depth.max()
        if d_max > d_min:
            self.normalized = (self.depth - d_min) / (d_max - d_min)
        else:
            self.normalized = torch.zeros_like(self.depth)
    
    def _compute_inverse(self) -> None:
        """Compute inverse depth."""
        eps = 1e-6
        self.inverse = 1.0 / (self.depth + eps)
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Get depth map shape (H, W)."""
        return self.depth.shape[-2:]
    
    def to(self, device: torch.device) -> "DepthMap":
        """Move depth map to device."""
        return DepthMap(
            depth=self.depth.to(device),
            normalized=self.normalized.to(device) if self.normalized is not None else None,
            inverse=self.inverse.to(device) if self.inverse is not None else None,
            scale=self.scale,
            intrinsics=self.intrinsics
        )
    
    def numpy(self) -> "DepthMap":
        """Convert to numpy arrays."""
        return DepthMap(
            depth=self.depth.cpu().numpy(),
            normalized=self.normalized.cpu().numpy() if self.normalized is not None else None,
            inverse=self.inverse.cpu().numpy() if self.inverse is not None else None,
            scale=self.scale,
            intrinsics=self.intrinsics
        )
    
    def get_confidence(self, threshold: float = 0.5) -> torch.Tensor:
        """Get confidence mask based on depth variance."""
        if self.normalized is None:
            self._normalize()
        return (self.normalized > threshold).float()
    
    def get_edges(self) -> torch.Tensor:
        """Extract depth edges using Sobel operator."""
        import kornia.filters as kf
        sobel_x = kf.sobel(self.depth.unsqueeze(0).unsqueeze(0), mode="diff")
        sobel_y = kf.sobel(self.depth.unsqueeze(0).unsqueeze(0), mode="diff")
        edges = torch.sqrt(sobel_x**2 + sobel_y**2).squeeze()
        return edges


@dataclass  
class NormalMap:
    """Container for surface normal estimation output.
    
    Attributes:
        normals: Surface normals as tensor [H, W, 3] or [3, H, W]
        confidence: Confidence scores for each normal vector
        reference_frame: Frame of reference ('camera' or 'world')
    """
    normals: torch.Tensor
    confidence: Optional[torch.Tensor] = None
    reference_frame: str = "camera"
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """Get normal map shape."""
        if self.normals.dim() == 4:
            return tuple(self.normals.shape[2:]) + (3,)
        return tuple(self.normals.shape)
    
    def normalize(self) -> None:
        """Ensure normals are unit vectors."""
        norm = torch.norm(self.normals, p=2, dim=-1, keepdim=True)
        self.normals = self.normals / (norm + 1e-8)
    
    def to(self, device: torch.device) -> "NormalMap":
        """Move normal map to device."""
        return NormalMap(
            normals=self.normals.to(device),
            confidence=self.confidence.to(device) if self.confidence is not None else None,
            reference_frame=self.reference_frame
        )
    
    def numpy(self) -> "NormalMap":
        """Convert to numpy arrays."""
        return NormalMap(
            normals=self.normals.cpu().numpy(),
            confidence=self.confidence.cpu().numpy() if self.confidence is not None else None,
            reference_frame=self.reference_frame
        )
    
    def visualize(self) -> np.ndarray:
        """Visualize normals as RGB image."""
        import numpy as np
        normals_vis = self.normals.cpu().numpy()
        if normals_vis.ndim == 4:
            normals_vis = normals_vis[0].transpose(1, 2, 0)
        elif normals_vis.shape[0] == 3:
            normals_vis = normals_vis.transpose(1, 2, 0)
        normals_vis = (normals_vis + 1) / 2
        return np.clip(normals_vis, 0, 1)