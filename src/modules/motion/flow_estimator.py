"""Optical flow and motion estimation using RAFT, GMFlow, and other models."""

from __future__ import annotations

from typing import List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .types import FlowField, MotionTrajectory, MotionMagnitude


class FlowEstimator:
    """Unified optical flow estimation interface.
    
    Supports:
    - RAFT: Recurrent All-Pairs Field Transforms (high accuracy)
    - GMFlow: Global Matching Flow (transformer-based)
    - Farneback: Polynomial expansion (CPU fallback)
    - Simple: Basic template matching fallback
    
    Args:
        config: Motion configuration object
        device: Target compute device
    """
    
    SUPPORTED_MODELS = ["raft", "gmflow", "farnback", "simple"]
    
    def __init__(
        self,
        config: Optional[object] = None,
        device: Optional[torch.device] = None
    ):
        self.config = config or self._default_config()
        self.device = device or torch.device("cpu")
        self.model = None
        self.model_type = getattr(self.config, 'model_type', 'raft').lower()
        self._initialized = False
        self._context = {}
    
    def _default_config(self):
        """Get default configuration."""
        class DefaultConfig:
            model_type = "raft"
            pretrained = True
            variant = "things"
            num_iters = 12
            context_frames = 2
        return DefaultConfig()
    
    def initialize(self) -> None:
        """Initialize the flow estimation model."""
        if self._initialized:
            return
        
        if self.model_type == "raft":
            self._init_raft()
        elif self.model_type == "gmflow":
            self._init_gmflow()
        else:
            self._init_fallback()
        
        self._initialized = True
    
    def _init_raft(self) -> None:
        """Initialize RAFT model."""
        try:
            from .raft_core.raft import RAFT
            from .raft_core.utils.utils import InputPadder
            
            self.raft = RAFT()
            
            checkpoint = getattr(self.config, 'checkpoint', None)
            if checkpoint is None or not torch.path.exists(checkpoint):
                checkpoint = self._download_raft_weights()
            
            state_dict = torch.load(checkpoint, map_location=self.device)
            
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            
            self.raft.load_state_dict(new_state_dict)
            self.raft = self.raft.to(self.device)
            self.raft.eval()
            
            self._padder_class = InputPadder
            
        except Exception as e:
            print(f"RAFT initialization failed: {e}. Using fallback.")
            self._init_fallback()
    
    def _download_raft_weights(self) -> str:
        """Download RAFT weights."""
        import os
        import urllib.request
        
        cache_dir = getattr(self.config, 'cache_dir', 'models/motion')
        os.makedirs(cache_dir, exist_ok=True)
        save_path = os.path.join(cache_dir, "raft-things.pth")
        
        if not os.path.exists(save_path):
            url = "https://dl.dropboxusercontent.com/s/4j84m2jsxmtpxlq/raft-things.pth"
            print(f"Downloading RAFT weights from {url}...")
            urllib.request.urlretrieve(url, save_path)
        
        return save_path
    
    def _init_gmflow(self) -> None:
        """Initialize GMFlow model."""
        try:
            from .gmflow.gmflow import GMFlow
            
            self.model = GMFlow(
                num_levels=4,
                feature_channels=128,
                attention_type="swin",
                num_transformer_layers=6,
            )
            
            checkpoint = getattr(self.config, 'checkpoint', None)
            if checkpoint is None:
                checkpoint = self._download_gmflow_weights()
            
            state_dict = torch.load(checkpoint, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)
            self.model = self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            print(f"GMFlow initialization failed: {e}. Using fallback.")
            self._init_fallback()
    
    def _download_gmflow_weights(self) -> str:
        """Download GMFlow weights."""
        import os
        import urllib.request
        
        cache_dir = getattr(self.config, 'cache_dir', 'models/motion')
        os.makedirs(cache_dir, exist_ok=True)
        save_path = os.path.join(cache_dir, "gmflow.pth")
        
        if not os.path.exists(save_path):
            url = "https://github.com/google-research/google-research/releases/download/gmflow/gmflow_sintel.pth"
            print(f"Downloading GMFlow weights from {url}...")
            urllib.request.urlretrieve(url, save_path)
        
        return save_path
    
    def _init_fallback(self) -> None:
        """Initialize fallback flow estimation."""
        self.model_type = "farnback"
        self.model = None
    
    def estimate(
        self,
        image1: Union[Image.Image, np.ndarray, torch.Tensor],
        image2: Union[Image.Image, np.ndarray, torch.Tensor],
        num_iters: Optional[int] = None,
        test_mode: bool = True
    ) -> FlowField:
        """Estimate optical flow between two images.
        
        Args:
            image1: First image (source)
            image2: Second image (target)
            num_iters: Number of refinement iterations
            test_mode: Use test mode (no dropout)
            
        Returns:
            FlowField object with flow vectors
        """
        if not self._initialized:
            self.initialize()
        
        img1 = self._preprocess_image(image1)
        img2 = self._preprocess_image(image2)
        
        if self.model_type == "raft":
            return self._estimate_raft(img1, img2, num_iters, test_mode)
        elif self.model_type == "gmflow":
            return self._estimate_gmflow(img1, img2)
        else:
            return self._estimate_farnback(image1, image2)
    
    def _preprocess_image(self, image: Union[Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Preprocess image to tensor."""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if isinstance(image, np.ndarray):
            if image.max() <= 1:
                image = (image * 255).astype(np.uint8)
            elif image.dtype != np.uint8:
                image = image.astype(np.uint8)
            
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            image = image.unsqueeze(0)
        
        return image.to(self.device)
    
    def _estimate_raft(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        num_iters: Optional[int],
        test_mode: bool
    ) -> FlowField:
        """Estimate flow using RAFT."""
        if num_iters is None:
            num_iters = getattr(self.config, 'num_iters', 12)
        
        h, w = image1.shape[2:]
        
        padder = self._padder_class(image1)
        image1, image2 = padder.pad(image1, image2)
        
        with torch.no_grad():
            flow_predictions = self.raft(
                image1,
                image2,
                iters=num_iters,
                test_mode=test_mode
            )
        
        flow = flow_predictions[-1]
        flow = padder.unpad(flow)
        flow = flow[0].permute(1, 2, 0).cpu().numpy()
        
        return FlowField(flow=flow)
    
    def _estimate_gmflow(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor
    ) -> FlowField:
        """Estimate flow using GMFlow."""
        h, w = image1.shape[2:]
        
        with torch.no_grad():
            flow = self.model(
                image1,
                image2,
                attn_splits_list=[[4, 4]],
                corr_radius=-1,
                prop_radius=18,
            )
        
        flow = flow[-1][0].permute(1, 2, 0).cpu().numpy()
        
        return FlowField(flow=flow)
    
    def _estimate_farnback(
        self,
        image1: Union[Image.Image, np.ndarray],
        image2: Union[Image.Image, np.ndarray]
    ) -> FlowField:
        """Estimate flow using Farneback algorithm."""
        import cv2
        
        if isinstance(image1, Image.Image):
            image1 = np.array(image1)
        if isinstance(image2, Image.Image):
            image2 = np.array(image2)
        
        if image1.ndim == 3:
            image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
        if image2.ndim == 3:
            image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
        
        if image1.max() > 1:
            image1 = (image1 / 255.0 * 255).astype(np.uint8)
            image2 = (image2 / 255.0 * 255).astype(np.uint8)
        
        flow = cv2.calcOpticalFlowFarneback(
            image1,
            image2,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
        return FlowField(flow=flow)
    
    def estimate_sequence(
        self,
        images: List[Union[Image.Image, np.ndarray]],
        bidirectional: bool = True
    ) -> List[FlowField]:
        """Estimate flow for a sequence of images.
        
        Args:
            images: List of images in sequence
            bidirectional: Estimate both forward and backward flow
            
        Returns:
            List of flow fields (one per pair)
        """
        flows = []
        
        for i in range(len(images) - 1):
            flow_fwd = self.estimate(images[i], images[i + 1])
            flows.append(flow_fwd)
            
            if bidirectional:
                flow_bwd = self.estimate(images[i + 1], images[i])
                flows.append(flow_bwd)
        
        return flows
    
    def compute_motion_mask(
        self,
        flow: FlowField,
        threshold: float = 1.0,
        erosion_size: int = 3
    ) -> np.ndarray:
        """Compute binary motion mask from flow.
        
        Args:
            flow: Flow field
            threshold: Motion magnitude threshold
            erosion_size: Size of erosion kernel for cleanup
            
        Returns:
            Binary motion mask
        """
        magnitude = flow.magnitude
        if isinstance(magnitude, torch.Tensor):
            magnitude = magnitude.cpu().numpy()
        
        mask = magnitude > threshold
        
        if erosion_size > 0:
            import cv2
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (erosion_size, erosion_size)
            )
            mask = cv2.erode(mask.astype(np.uint8), kernel)
            mask = cv2.dilate(mask, kernel)
        
        return mask
    
    def get_motion_statistics(self, flow: FlowField) -> MotionMagnitude:
        """Get motion statistics from flow field."""
        return MotionMagnitude.from_flow(flow.flow)
    
    def track_points(
        self,
        points: np.ndarray,
        flow: FlowField,
        search_window: int = 30
    ) -> np.ndarray:
        """Track points using flow field.
        
        Args:
            points: Initial points to track [N, 2] (x, y)
            flow: Flow field
            search_window: Search window size for tracking
            
        Returns:
            Tracked points [N, 2]
        """
        if points.ndim == 1:
            points = points.reshape(-1, 2)
        
        h, w = flow.shape
        tracked = np.zeros_like(points)
        
        for i, (x, y) in enumerate(points):
            x, y = int(x), int(y)
            x = min(max(x, 0), w - 1)
            y = min(max(y, 0), h - 1)
            
            if isinstance(flow.flow, torch.Tensor):
                flow_np = flow.flow.cpu().numpy()
            else:
                flow_np = flow.flow
            
            if flow_np.ndim == 3:
                dx, dy = flow_np[y, x]
            else:
                dx, dy = flow_np[0, y, x], flow_np[1, y, x]
            
            new_x = min(max(x + dx, 0), w - 1)
            new_y = min(max(y + dy, 0), h - 1)
            
            tracked[i] = [new_x, new_y]
        
        return tracked
    
    def create_trajectory(
        self,
        initial_points: np.ndarray,
        flows: List[FlowField]
    ) -> List[MotionTrajectory]:
        """Create motion trajectories from a sequence of flow fields.
        
        Args:
            initial_points: Starting points [N, 2]
            flows: List of flow fields
            
        Returns:
            List of trajectories
        """
        trajectories = []
        
        for i, point in enumerate(initial_points):
            traj = MotionTrajectory(
                points=[tuple(point)],
                track_id=i
            )
            
            current_point = point.copy()
            for flow in flows:
                tracked = self.track_points(current_point.reshape(1, 2), flow)
                current_point = tracked[0]
                traj.points.append(tuple(current_point))
            
            trajectories.append(traj)
        
        return trajectories
    
    def __repr__(self) -> str:
        return f"FlowEstimator(model={self.model_type}, device={self.device})"


class RAFTConv(nn.Module):
    """RAFT convolution module."""
    
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class BasicEncoder(nn.Module):
    """Basic encoder for optical flow models."""
    
    def __init__(self, output_dim=128):
        super().__init__()
        self.conv1 = RAFTConv(3, 64, kernel_size=7)
        self.conv2 = RAFTConv(64, 96, kernel_size=3)
        self.conv3 = RAFTConv(96, 128, kernel_size=3)
        self.conv4 = RAFTConv(128, 96, kernel_size=3)
        self.conv5 = RAFTConv(96, output_dim, kernel_size=3)
        
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x1 = self.conv1(x)
        x = self.pool(x1)
        x2 = self.conv2(x)
        x = self.pool(x2)
        x3 = self.conv3(x)
        x = self.conv4(x3)
        x = self.conv5(x)
        
        return x, x1, x2, x3