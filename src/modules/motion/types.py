"""Type definitions for motion estimation."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np
import torch


@dataclass
class FlowField:
    """Optical flow field container.
    
    Attributes:
        flow: Flow vectors [H, W, 2] (u, v) or [2, H, W]
        magnitude: Flow magnitude per pixel
        angle: Flow angle per pixel (radians)
        confidence: Confidence scores for each flow vector
        timestamp: Timestamp for this flow field
    """
    flow: Union[np.ndarray, torch.Tensor]
    magnitude: Optional[Union[np.ndarray, torch.Tensor]] = None
    angle: Optional[Union[np.ndarray, torch.Tensor]] = None
    confidence: Optional[Union[np.ndarray, torch.Tensor]] = None
    timestamp: Optional[float] = None
    
    def __post_init__(self):
        if self.magnitude is None:
            self._compute_magnitude()
        if self.angle is None:
            self._compute_angle()
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Get flow field shape."""
        if isinstance(self.flow, np.ndarray):
            if self.flow.ndim == 3:
                return (self.flow.shape[0], self.flow.shape[1])
            return self.flow.shape[:2]
        else:
            if self.flow.dim() == 3:
                return (self.flow.shape[1], self.flow.shape[2])
            return self.flow.shape[:2]
    
    def _compute_magnitude(self) -> None:
        """Compute flow magnitude."""
        if isinstance(self.flow, np.ndarray):
            if self.flow.ndim == 3:
                flow = self.flow.transpose(2, 0, 1)
            else:
                flow = self.flow
            self.magnitude = np.sqrt(np.sum(flow ** 2, axis=0))
        else:
            if self.flow.dim() == 3:
                flow = self.flow
            else:
                flow = self.flow.unsqueeze(0)
            self.magnitude = torch.sqrt(torch.sum(flow ** 2, dim=0))
    
    def _compute_angle(self) -> None:
        """Compute flow angle in radians."""
        if isinstance(self.flow, np.ndarray):
            if self.flow.ndim == 3:
                u, v = self.flow[0], self.flow[1]
            else:
                u, v = self.flow[..., 0], self.flow[..., 1]
            self.angle = np.arctan2(v, u)
        else:
            if self.flow.dim() == 3:
                u = self.flow[0]
                v = self.flow[1]
            else:
                u = self.flow[..., 0]
                v = self.flow[..., 1]
            self.angle = torch.atan2(v, u)
    
    def to(self, device: torch.device) -> "FlowField":
        """Move flow field to device."""
        return FlowField(
            flow=self.flow.to(device) if isinstance(self.flow, torch.Tensor) else torch.from_numpy(self.flow).to(device),
            magnitude=self.magnitude.to(device) if self.magnitude is not None and isinstance(self.magnitude, torch.Tensor) else self.magnitude,
            angle=self.angle.to(device) if self.angle is not None and isinstance(self.angle, torch.Tensor) else self.angle,
            confidence=self.confidence.to(device) if self.confidence is not None and isinstance(self.confidence, torch.Tensor) else self.confidence,
            timestamp=self.timestamp
        )
    
    def numpy(self) -> "FlowField":
        """Convert to numpy arrays."""
        return FlowField(
            flow=self.flow.cpu().numpy() if isinstance(self.flow, torch.Tensor) else self.flow,
            magnitude=self.magnitude.cpu().numpy() if isinstance(self.magnitude, torch.Tensor) else self.magnitude,
            angle=self.angle.cpu().numpy() if isinstance(self.angle, torch.Tensor) else self.angle,
            confidence=self.confidence.cpu().numpy() if isinstance(self.confidence, torch.Tensor) else self.confidence,
            timestamp=self.timestamp
        )
    
    def visualize(self, max_flow: Optional[float] = None) -> np.ndarray:
        """Visualize flow as RGB image using color wheel.
        
        Args:
            max_flow: Maximum flow magnitude for normalization
            
        Returns:
            RGB image visualization
        """
        import cv2
        
        flow_np = self.flow
        if isinstance(flow_np, torch.Tensor):
            flow_np = flow_np.cpu().numpy()
        
        if flow_np.ndim == 3:
            flow_np = flow_np.transpose(1, 2, 0)
        
        magnitude = self.magnitude
        if isinstance(magnitude, torch.Tensor):
            magnitude = magnitude.cpu().numpy()
        
        if max_flow is None:
            max_flow = np.percentile(magnitude, 95) if magnitude is not None else 1.0
            max_flow = max(max_flow, 1e-6)
        
        h, w = flow_np.shape[:2]
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        
        if magnitude is not None:
            hsv[..., 2] = np.clip(magnitude / max_flow * 255, 0, 255).astype(np.uint8)
        else:
            mag = np.sqrt(flow_np[..., 0]**2 + flow_np[..., 1]**2)
            hsv[..., 2] = np.clip(mag / max_flow * 255, 0, 255).astype(np.uint8)
        
        angle = self.angle
        if isinstance(angle, torch.Tensor):
            angle = angle.cpu().numpy()
        if angle is None:
            angle = np.arctan2(flow_np[..., 1], flow_np[..., 0])
        
        hsv[..., 0] = ((angle / (2 * np.pi) + 1) % 1 * 180).astype(np.uint8)
        hsv[..., 1] = 255
        
        rgb = cv2.cvtColor(hsv, cv2.COLOR__HSV2BGR if hasattr(cv2, 'COLOR_HSV2BGR') else cv2.COLOR_HSV2BGR)
        return rgb
    
    def warp_image(self, image: np.ndarray) -> np.ndarray:
        """Warp image using flow field.
        
        Args:
            image: Image to warp [H, W, C]
            
        Returns:
            Warped image
        """
        import cv2
        
        flow_np = self.flow
        if isinstance(flow_np, torch.Tensor):
            flow_np = flow_np.cpu().numpy()
        
        if flow_np.ndim == 3:
            flow_np = flow_np.transpose(1, 2, 0)
        
        h, w = image.shape[:2]
        flow_map = flow_np.copy()
        flow_map[..., 0] += np.arange(w)
        flow_map[..., 1] += np.arange(h)[:, np.newaxis]
        
        warped = cv2.remap(
            image,
            flow_map,
            None,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )
        
        return warped
    
    def get_motion_boundaries(self, threshold: float = 1.0) -> np.ndarray:
        """Get motion boundaries using gradient magnitude."""
        import cv2
        
        magnitude = self.magnitude
        if isinstance(magnitude, torch.Tensor):
            magnitude = magnitude.cpu().numpy()
        
        if magnitude is None:
            flow_np = self.flow
            if isinstance(flow_np, torch.Tensor):
                flow_np = flow_np.cpu().numpy()
            magnitude = np.sqrt(flow_np[0]**2 + flow_np[1]**2)
        
        grad_x = cv2.Sobel(magnitude, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(magnitude, cv2.CV_32F, 0, 1, ksize=3)
        boundary = np.sqrt(grad_x**2 + grad_y**2)
        
        return (boundary > threshold).astype(np.uint8)


@dataclass
class MotionTrajectory:
    """Container for motion trajectory of tracked points.
    
    Attributes:
        points: List of point positions over time [(x, y), ...]
        velocity: Velocity vectors at each point
        acceleration: Acceleration vectors at each point
        timestamps: Timestamps for each frame
        track_id: Unique identifier for this trajectory
        label: Semantic label for the trajectory
    """
    points: List[Tuple[float, float]] = field(default_factory=list)
    velocity: Optional[List[Tuple[float, float]]] = None
    acceleration: Optional[List[Tuple[float, float]]] = None
    timestamps: Optional[List[float]] = None
    track_id: Optional[int] = None
    label: Optional[str] = None
    
    @property
    def length(self) -> int:
        return len(self.points)
    
    @property
    def total_distance(self) -> float:
        if len(self.points) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(self.points)):
            dx = self.points[i][0] - self.points[i-1][0]
            dy = self.points[i][1] - self.points[i-1][1]
            total += np.sqrt(dx**2 + dy**2)
        return total
    
    def compute_velocity(self, dt: float = 1.0) -> None:
        """Compute velocity from positions."""
        if len(self.points) < 2:
            return
        
        self.velocity = []
        for i in range(len(self.points)):
            if i == 0:
                dx = self.points[1][0] - self.points[0][0]
                dy = self.points[1][1] - self.points[0][1]
            elif i == len(self.points) - 1:
                dx = self.points[-1][0] - self.points[-2][0]
                dy = self.points[-1][1] - self.points[-2][1]
            else:
                dx = (self.points[i+1][0] - self.points[i-1][0]) / 2
                dy = (self.points[i+1][1] - self.points[i-1][1]) / 2
            
            speed = np.sqrt(dx**2 + dy**2) / dt
            angle = np.arctan2(dy, dx)
            self.velocity.append((speed, angle))
    
    def compute_acceleration(self, dt: float = 1.0) -> None:
        """Compute acceleration from velocities."""
        if self.velocity is None or len(self.velocity) < 2:
            self.compute_velocity(dt)
        
        if self.velocity is None or len(self.velocity) < 2:
            return
        
        self.acceleration = []
        for i in range(len(self.velocity)):
            if i == 0:
                dv_s = self.velocity[1][0] - self.velocity[0][0]
                dv_a = self.velocity[1][1] - self.velocity[0][1]
            elif i == len(self.velocity) - 1:
                dv_s = self.velocity[-1][0] - self.velocity[-2][0]
                dv_a = self.velocity[-1][1] - self.velocity[-2][1]
            else:
                dv_s = (self.velocity[i+1][0] - self.velocity[i-1][0]) / 2
                dv_a = (self.velocity[i+1][1] - self.velocity[i-1][1]) / 2
            
            self.acceleration.append((dv_s / dt, dv_a / dt))
    
    def predict_next(self, frames: int = 1) -> List[Tuple[float, float]]:
        """Predict future positions using velocity."""
        if self.velocity is None:
            self.compute_velocity()
        
        if self.velocity is None:
            return []
        
        predictions = []
        last_point = self.points[-1] if self.points else (0, 0)
        last_vel = self.velocity[-1] if self.velocity else (0, 0)
        
        speed, angle = last_vel
        vx = speed * np.cos(angle)
        vy = speed * np.sin(angle)
        
        for _ in range(frames):
            new_x = last_point[0] + vx
            new_y = last_point[1] + vy
            predictions.append((new_x, new_y))
            last_point = (new_x, new_y)
        
        return predictions
    
    def get_smoothed(self, window_size: int = 3) -> List[Tuple[float, float]]:
        """Apply smoothing to trajectory."""
        if len(self.points) < window_size:
            return self.points
        
        smoothed = []
        half_w = window_size // 2
        
        for i in range(len(self.points)):
            start = max(0, i - half_w)
            end = min(len(self.points), i + half_w + 1)
            window = self.points[start:end]
            smoothed.append((
                np.mean([p[0] for p in window]),
                np.mean([p[1] for p in window])
            ))
        
        return smoothed


@dataclass
class MotionMagnitude:
    """Motion magnitude statistics for a region or image.
    
    Attributes:
        mean: Mean motion magnitude
        median: Median motion magnitude
        max: Maximum motion magnitude
        std: Standard deviation of motion magnitude
        histogram: Optional histogram of magnitudes
        bins: Bin edges for histogram
    """
    mean: float
    median: float
    max: float
    std: float
    histogram: Optional[np.ndarray] = None
    bins: Optional[np.ndarray] = None
    
    @classmethod
    def from_flow(cls, flow: Union[np.ndarray, torch.Tensor]) -> "MotionMagnitude":
        """Compute statistics from flow field."""
        if isinstance(flow, torch.Tensor):
            flow = flow.cpu().numpy()
        
        if flow.ndim == 3:
            flow = flow.transpose(1, 2, 0)
        
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        
        hist, bins = np.histogram(magnitude.flatten(), bins=50)
        
        return cls(
            mean=float(magnitude.mean()),
            median=float(np.median(magnitude)),
            max=float(magnitude.max()),
            std=float(magnitude.std()),
            histogram=hist,
            bins=bins
        )
    
    def __repr__(self) -> str:
        return f"MotionMagnitude(mean={self.mean:.3f}, max={self.max:.3f})"