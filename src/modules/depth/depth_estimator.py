"""Depth estimation using state-of-the-art models (MiDaS, ZoeDepth, Marigold)."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .types import DepthMap, NormalMap


class DepthEstimator:
    """Unified depth estimation interface supporting multiple backends.
    
    Supports:
    - ZoeDepth: Metric depth estimation (best for indoor/outdoor scenes)
    - MiDaS: Monocular depth estimation (generic)
    - Marigold: Monocular depth regression (diffusion-based)
    
    Args:
        config: Depth configuration object
        device: Target compute device
    """
    
    SUPPORTED_MODELS = ["zoedepth", "midas", "marigold"]
    
    def __init__(
        self,
        config: Optional[object] = None,
        device: Optional[torch.device] = None
    ):
        self.config = config or self._default_config()
        self.device = device or torch.device("cpu")
        self.model = None
        self.model_type = self.config.name.lower() if hasattr(self.config, 'name') else "zoedepth"
        self._initialized = False
    
    def _default_config(self):
        """Get default configuration."""
        class DefaultConfig:
            name = "Zoedepth"
            model_type = "ZoeD"
            variant = "kitti"
            pretrained = True
        return DefaultConfig()
    
    def initialize(self) -> None:
        """Initialize the depth estimation model."""
        if self._initialized:
            return
        
        if self.model_type == "zoedepth":
            self._init_zoedepth()
        elif self.model_type == "midas":
            self._init_midas()
        elif self.model_type == "marigold":
            self._init_marigold()
        else:
            self._init_zoedepth()
        
        self._initialized = True
    
    def _init_zoedepth(self) -> None:
        """Initialize ZoeDepth model."""
        try:
            from zoedepth.models.builder import build_model
            from zoedepth.utils.config import get_config
            
            variant = getattr(self.config, 'variant', 'kitti')
            
            try:
                config = get_config("zoedepth", "infer")
                config.models['depth']['pretrained'] = "home/dh/.cache/zoedepth/weights"
                self.model = build_model(config)
            except Exception:
                try:
                    from zoedepth.utils.misc import get_pretrained_url
                    pretrained_url = get_pretrained_url("ZoeD", variant)
                    self.model = build_model(config, pretrained_url=pretrained_url)
                except Exception:
                    from zoedepth.models import ZoeDepth
                    self.model = ZoeDepth.from_pretrained(f"Zoedepth/{variant}")
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
        except ImportError:
            self.model = self._create_fallback_depth_model()
    
    def _init_midas(self) -> None:
        """Initialize MiDaS model."""
        try:
            from transformers import DPTImageProcessor, DPTForDepthEstimation
            
            self.processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
            self.model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas")
            self.model = self.model.to(self.device)
            self.model.eval()
            
        except ImportError:
            self.model = self._create_fallback_depth_model()
    
    def _init_marigold(self) -> None:
        """Initialize Marigold diffusion model."""
        try:
            from diffusers import DiffusionPipeline
            
            self.model = DiffusionPipeline.from_pretrained(
                "prs-eth/marigold-depth",
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            
        except ImportError:
            self.model = self._create_fallback_depth_model()
    
    def _create_fallback_depth_model(self) -> nn.Module:
        """Create a lightweight fallback depth estimation model."""
        class SimpleDepthNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                )
                self.decoder = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                    nn.Conv2d(128, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 32, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                    nn.Conv2d(32, 16, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(16, 1, 1),
                )
            
            def forward(self, x):
                features = self.encoder(x)
                depth = self.decoder(features)
                return depth
            
            def predict(self, image):
                with torch.no_grad():
                    img_tensor = self.preprocess_image(image)
                    depth = self.forward(img_tensor)
                    depth = F.interpolate(
                        depth.unsqueeze(0),
                        size=(image.height, image.width),
                        mode="bilinear",
                        align_corners=False
                    )
                    return depth.squeeze(0).squeeze(0)
            
            @staticmethod
            def preprocess_image(image):
                if isinstance(image, Image.Image):
                    image = np.array(image)
                if image.ndim == 2:
                    image = np.stack([image] * 3, axis=-1)
                if image.max() > 1:
                    image = image.astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
                return img_tensor.unsqueeze(0)
        
        model = SimpleDepthNet()
        model = model.to(self.device)
        model.eval()
        return model
    
    def estimate(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        normalize: bool = True,
        return_normal: bool = False
    ) -> Union[DepthMap, Tuple[DepthMap, NormalMap]]:
        """Estimate depth from an input image.
        
        Args:
            image: Input image (PIL Image, numpy array, or tensor)
            normalize: Whether to normalize output to [0, 1]
            return_normal: Whether to also return surface normals
            
        Returns:
            DepthMap object, optionally with NormalMap
        """
        if not self._initialized:
            self.initialize()
        
        if isinstance(image, Image.Image):
            image = np.array(image)
        if isinstance(image, np.ndarray):
            image = self._preprocess_image(image)
        
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        if image.max() > 1:
            image = image / 255.0
        
        image = image.to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            if self.model_type == "zoedepth":
                depth = self._estimate_zoedepth(image)
            elif self.model_type == "midas":
                depth = self._estimate_midas(image)
            elif self.model_type == "marigold":
                depth = self._estimate_marigold(image)
            else:
                depth = self.model(image)
        
        depth_map = DepthMap(
            depth=depth.squeeze(),
            scale=1.0
        )
        
        if normalize:
            depth_min = depth.min()
            depth_max = depth.max()
            if depth_max > depth_min:
                depth_map.normalized = (depth - depth_min) / (depth_max - depth_min)
            else:
                depth_map.normalized = torch.zeros_like(depth)
        
        if return_normal:
            normal_map = self._compute_normals(depth)
            return depth_map, normal_map
        
        return depth_map
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for depth estimation."""
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        if image.shape[-1] == 4:
            image = image[:, :, :3]
        
        return image
    
    def _estimate_zoedepth(self, image: torch.Tensor) -> torch.Tensor:
        """Estimate depth using ZoeDepth."""
        try:
            depth = self.model.infer(image)
            return depth
        except Exception:
            original_size = image.shape[-2:]
            output = self.model(image)
            if isinstance(output, dict):
                depth = output.get("depth", output.get("out", output.get("pred", output)))
            else:
                depth = output
            depth = F.interpolate(
                depth,
                size=original_size,
                mode="bilinear",
                align_corners=False
            )
            return depth.squeeze(0).squeeze(0)
    
    def _estimate_midas(self, image: torch.Tensor) -> torch.Tensor:
        """Estimate depth using MiDaS."""
        original_size = image.shape[-2:]
        
        if hasattr(self, 'processor'):
            inputs = self.processor(images=image.squeeze(0).permute(1, 2, 0).cpu().numpy(), return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            depth = outputs.predicted_depth
        else:
            depth = self.model(image)
        
        depth = F.interpolate(
            depth.unsqueeze(0),
            size=original_size,
            mode="bilinear",
            align_corners=False
        )
        return depth.squeeze(0).squeeze(0)
    
    def _estimate_marigold(self, image: torch.Tensor) -> torch.Tensor:
        """Estimate depth using Marigold."""
        original_size = image.shape[-2:]
        rgb = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        rgb = (np.clip(rgb, 0, 1) * 65535).astype(np.uint16)
        
        with torch.no_grad():
            depth = self.model.predict_legacy(rgb, denormalize=False)
        
        depth = torch.from_numpy(depth).float().to(self.device)
        depth = F.interpolate(
            depth.unsqueeze(0).unsqueeze(0),
            size=original_size,
            mode="bilinear",
            align_corners=False
        )
        return depth.squeeze(0).squeeze(0)
    
    def _compute_normals(self, depth: torch.Tensor) -> NormalMap:
        """Compute surface normals from depth map."""
        if depth.dim() == 2:
            depth = depth.unsqueeze(0).unsqueeze(0)
        elif depth.dim() == 3:
            depth = depth.unsqueeze(0)
        
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=depth.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=depth.device)
        
        pad = 1
        depth_padded = F.pad(depth, (pad, pad, pad, pad), mode='replicate')
        
        dx = F.conv2d(depth_padded, sobel_x.unsqueeze(0).unsqueeze(0))
        dy = F.conv2d(depth_padded, sobel_y.unsqueeze(0).unsqueeze(0))
        
        normal_x = -dx
        normal_y = -dy
        normal_z = torch.ones_like(dx)
        
        normals = torch.cat([normal_x, normal_y, normal_z], dim=1)
        normals = F.normalize(normals, p=2, dim=1)
        
        return NormalMap(normals=normals.squeeze(0))
    
    def estimate_batch(
        self,
        images: list,
        batch_size: int = 4
    ) -> list[DepthMap]:
        """Estimate depth for a batch of images.
        
        Args:
            images: List of images
            batch_size: Batch size for processing
            
        Returns:
            List of DepthMap objects
        """
        if not self._initialized:
            self.initialize()
        
        results = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            tensors = []
            for img in batch:
                if isinstance(img, Image.Image):
                    img = np.array(img)
                if isinstance(img, np.ndarray):
                    img = self._preprocess_image(img)
                    img = torch.from_numpy(img).permute(2, 0, 1).float()
                tensors.append(img)
            
            batch_tensor = torch.stack(tensors).to(self.device)
            
            with torch.no_grad():
                depths = self.model.infer(batch_tensor) if self.model_type == "zoedepth" else self.model(batch_tensor)
            
            for j, depth in enumerate(depths if hasattr(depths, '__iter__') else [depths]):
                if hasattr(depth, '__iter__'):
                    depth_j = list(depths)[j] if j < len(list(depths)) else list(depths)[0]
                else:
                    depth_j = depth[j] if depth.dim() > 0 else depth
                results.append(DepthMap(depth=depth_j))
        
        return results
    
    def load_weights(self, path: str) -> None:
        """Load model weights from checkpoint."""
        if self.model is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
    
    def save_weights(self, path: str) -> None:
        """Save model weights to checkpoint."""
        if self.model is None:
            raise RuntimeError("Model not initialized.")
        
        torch.save(self.model.state_dict(), path)
    
    def __repr__(self) -> str:
        return f"DepthEstimator(model={self.model_type}, device={self.device})"