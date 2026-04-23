"""Configuration management for Image2Video AI system."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml
from omegaconf import DictConfig, OmegaConf


@dataclass
class SystemConfig:
    """System-level configuration."""
    device: str = "cuda"
    mixed_precision: bool = True
    compile_model: bool = False
    
    def __post_init__(self):
        if self.device == "cuda" and not self._cuda_available():
            self.device = "cpu"
    
    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False


@dataclass
class DepthConfig:
    """Depth estimation model configuration."""
    name: str = "Zoedepth"
    model_type: str = "ZoeD"
    pretrained: bool = True
    variant: str = "kitti"
    device: str = "cuda"
    cache_dir: str = "models/depth"
    model_path: Optional[str] = None


@dataclass
class SegmentationConfig:
    """Segmentation model configuration."""
    name: str = "SAM"
    model_type: str = "vit_h"
    checkpoint: str = "sam_vit_h.pth"
    device: str = "cuda"
    cache_dir: str = "models/segmentation"
    points_per_side: int = 32
    pred_iou_thresh: float = 0.86
    stability_score_thresh: float = 0.92


@dataclass
class MotionConfig:
    """Motion estimation model configuration."""
    model_type: str = "raft"
    pretrained: bool = True
    variant: str = "things"
    device: str = "cuda"
    cache_dir: str = "models/motion"
    num_iters: int = 12
    context_frames: int = 2


@dataclass
class GenerationConfig:
    """Video generation model configuration."""
    name: str = "AnimateDiff"
    model_id: str = "stabilityai/stable-diffusion-xl-refiner-1.0"
    motion_adapter: str = "guoyww/animatediff-motion-adapter-sdxl-beta"
    device: str = "cuda"
    guidance_scale: float = 7.5
    num_inference_steps: int = 25
    fps: int = 8
    num_frames: int = 24
    prompt: str = ""
    negative_prompt: str = "blurry, low quality, artifacts, flickering"
    seed: Optional[int] = None


@dataclass
class ProcessingConfig:
    """Image processing configuration."""
    target_resolution: Tuple[int, int] = (512, 512)
    color_space: str = "RGB"
    normalize: bool = True
    center_crop: bool = True
    preserve_aspect_ratio: bool = True
    interpolation: str = "lanczos"


@dataclass
class OutputConfig:
    """Output configuration."""
    format: str = "mp4"
    codec: str = "libx264"
    crf: int = 23
    preset: str = "medium"
    audio: bool = False
    metadata: bool = True
    output_dir: str = "outputs"


class Config:
    """Main configuration class that loads and manages all settings."""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None
    ):
        self.config_path = config_path
        self._base_config = self._default_config()
        
        if config_path and os.path.exists(config_path):
            self._load_from_file(config_path)
        
        if overrides:
            self._apply_overrides(overrides)
        
        self._resolve_cache_dirs()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "system": SystemConfig(),
            "models": {
                "depth": DepthConfig(),
                "segmentation": SegmentationConfig(),
                "motion": MotionConfig(),
                "generation": GenerationConfig(),
            },
            "processing": ProcessingConfig(),
            "output": OutputConfig(),
        }
    
    def _load_from_file(self, path: str) -> None:
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            file_config = yaml.safe_load(f)
        
        if file_config:
            self._merge_config(file_config)
    
    def _merge_config(self, new_config: Dict[str, Any]) -> None:
        """Merge new configuration into base config."""
        for key, value in new_config.items():
            if key in self._base_config:
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if hasattr(self._base_config[key], sub_key):
                            setattr(self._base_config[key], sub_key, sub_value)
                elif hasattr(self._base_config[key], key):
                    setattr(self._base_config[key], key, value)
    
    def _apply_overrides(self, overrides: Dict[str, Any]) -> None:
        """Apply command-line overrides to configuration."""
        for key, value in overrides.items():
            keys = key.split(".")
            if len(keys) >= 2:
                section = keys[0]
                attr = keys[1]
                if section in self._base_config and hasattr(self._base_config[section], attr):
                    setattr(self._base_config[section], attr, value)
    
    def _resolve_cache_dirs(self) -> None:
        """Ensure cache directories exist and are accessible."""
        cache_dirs = [
            self._base_config["models"]["depth"].cache_dir,
            self._base_config["models"]["segmentation"].cache_dir,
            self._base_config["models"]["motion"].cache_dir,
        ]
        for cache_dir in cache_dirs:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    def __getattr__(self, name: str) -> Any:
        """Get configuration section by name."""
        if name in self._base_config:
            return self._base_config[name]
        raise AttributeError(f"Config has no section: {name}")
    
    @property
    def system(self) -> SystemConfig:
        return self._base_config["system"]
    
    @property
    def models(self) -> Dict[str, Any]:
        return self._base_config["models"]
    
    @property
    def processing(self) -> ProcessingConfig:
        return self._base_config["processing"]
    
    @property
    def output(self) -> OutputConfig:
        return self._base_config["output"]
    
    def get_model_config(self, model_name: str) -> Any:
        """Get configuration for a specific model."""
        return self._base_config["models"].get(model_name)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for key, value in self._base_config.items():
            if hasattr(value, "__dict__"):
                result[key] = vars(value)
            else:
                result[key] = value
        return result
    
    def save(self, path: str) -> None:
        """Save current configuration to file."""
        config_dict = self.to_dict()
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Create Config from YAML file."""
        return cls(config_path=path)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create Config from dictionary."""
        cfg = cls()
        cfg._merge_config(config_dict)
        return cfg


def load_config(path: str, **overrides) -> Config:
    """Load configuration from file with optional overrides."""
    return Config(config_path=path, overrides=overrides)