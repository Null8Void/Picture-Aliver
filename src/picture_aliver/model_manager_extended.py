"""
Picture-Aliver Extended Model Manager

Manages multiple video generation models with automatic fallback:
- Wan 2.1, Wan 2.2
- LightX2V framework
- HunyuanVideo
- LTX-Video
- CogVideo
- LongCat-Video
- Legacy pipeline

Usage:
    from src.picture_aliver.model_manager import ModelManager
    
    manager = ModelManager(primary="wan21", fallback=["lightx2v", "legacy"])
    result = manager.generate("input.jpg", "motion")
"""

from __future__ import annotations

import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
from dataclasses import dataclass, field

import torch

logger = logging.getLogger("picture_aliver.model_manager")


@dataclass
class ModelAttempt:
    """Result of a model attempt."""
    model_type: str
    success: bool
    error: Optional[str] = None
    generation_time: Optional[float] = None
    vram_required: Optional[float] = None


@dataclass
class ModelInfo:
    """Information about a model."""
    name: str
    display_name: str
    license: str
    min_vram_gb: float
    supports_i2v: bool
    supports_t2v: bool
    quality: str  # low, medium, high
    speed: str    # slow, medium, fast
    

# Model metadata
MODEL_INFO = {
    "wan21": ModelInfo("wan21", "Wan 2.1", "Apache 2.0", 16, True, True, "high", "medium"),
    "wan22": ModelInfo("wan22", "Wan 2.2", "Apache 2.0", 16, True, True, "high", "medium"),
    "lightx2v": ModelInfo("lightx2v", "LightX2V", "Apache 2.0", 8, True, True, "high", "fast"),
    "hunyuan": ModelInfo("hunyuan", "HunyuanVideo", "Research", 16, True, True, "high", "slow"),
    "ltx": ModelInfo("ltx", "LTX-Video", "Apache 2.0", 8, True, True, "medium", "fast"),
    "cogvideo": ModelInfo("cogvideo", "CogVideo", "Apache 2.0", 12, True, True, "medium", "slow"),
    "longcat": ModelInfo("longcat", "LongCat-Video", "MIT", 16, True, True, "high", "medium"),
    "legacy": ModelInfo("legacy", "Legacy Pipeline", "Apache 2.0", 2, True, True, "low", "slow"),
}


class ModelManager:
    """
    Manages video generation with automatic failover.
    
    Tries models in order until one succeeds.
    """
    
    def __init__(
        self,
        primary: str = "wan21",
        fallback: Optional[List[str]] = None,
        config_path: Optional[str] = None,
    ):
        """
        Initialize ModelManager.
        
        Args:
            primary: Primary model type
            fallback: List of fallback model types
            config_path: Path to config file
        """
        self.primary = primary
        self.fallback = fallback or ["legacy"]
        self.config_path = config_path
        
        self._current_model = None
        self._model_order: List[str] = []
        self._attempts: List[ModelAttempt] = []
        
        # Initialize logger first
        self._logger = logging.getLogger("picture_aliver.model_manager")
        
        # Then build model order
        self._build_model_order()
        
        self._logger = logging.getLogger("picture_aliver.model_manager")
        
    def _build_model_order(self):
        """Build ordered list of models to try."""
        # Check available VRAM
        vram_available = 0
        if torch.cuda.is_available():
            vram_available = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        self._logger.info(f"Available VRAM: {vram_available:.1f}GB")
        
        # Check available models (for CPU/low-VRAM, include legacy)
        available = self._check_available_models()
        
        # If nothing available, try all models anyway (user may have deps not detected)
        if not any(available.values()):
            self._logger.warning("No models detected as available, will try anyway")
            available = {k: True for k in MODEL_INFO}
        
        self._model_order = []
        
        # Add primary if available
        if self.primary in available and available[self.primary]:
            if self._can_use_model(self.primary, vram_available):
                self._model_order.append(self.primary)
        
        # Add fallbacks in order
        for model in self.fallback:
            if model not in self._model_order:
                if model in available and available[model]:
                    if self._can_use_model(model, vram_available):
                        self._model_order.append(model)
        
        # Add any other available models (legacy always last)
        for model in MODEL_INFO:
            if model not in self._model_order:
                if model in available and available[model]:
                    # Legacy always goes last
                    if model == "legacy":
                        continue
                    if self._can_use_model(model, vram_available):
                        self._model_order.append(model)
        
        # Add legacy at end if available
        if "legacy" in available and available["legacy"]:
            if "legacy" not in self._model_order:
                self._model_order.append("legacy")
        
        if not self._model_order:
            self._logger.error("No video generation models available!")
        
        self._logger.info(f"Model order: {self._model_order}")
    
    def _can_use_model(self, model_type: str, vram_available: float) -> bool:
        """Check if model can be used with available VRAM."""
        if model_type in MODEL_INFO:
            return MODEL_INFO[model_type].min_vram_gb <= vram_available
        return True
    
    def _check_available_models(self) -> Dict[str, bool]:
        """Check which models can be loaded."""
        available = {}
        
        # Check each model
        for model in MODEL_INFO:
            available[model] = self._check_model_availability(model)
        
        return available
    
    def _check_model_availability(self, model_type: str) -> bool:
        """Check if a specific model can be loaded."""
        try:
            if model_type == "legacy":
                project_root = Path(__file__).parent.parent.parent
                if str(project_root) not in sys.path:
                    sys.path.insert(0, str(project_root))
                from src.picture_aliver.main import Pipeline
                return True
            
            # Try importing model-specific loader
            if model_type in ("wan21", "wan22"):
                from diffusers import WanImageToVideoPipeline
                return True
            elif model_type == "lightx2v":
                from lightx2v import LightX2VPipeline
                return True
            elif model_type == "hunyuan":
                from diffusers import HunyuanVideoPipeline
                return True
            elif model_type == "ltx":
                from diffusers import LTXVideoPipeline
                return True
            elif model_type in ("cogvideo", "longcat"):
                return True
            
        except ImportError:
            pass
        except Exception as e:
            self._logger.debug(f"{model_type} check: {e}")
        
        return False
    
    def load_model(self, model_type: Optional[str] = None) -> Any:
        """Load a model."""
        if model_type is None:
            model_type = self._model_order[0] if self._model_order else None
        
        if model_type is None:
            return None
        
        self._logger.info(f"Loading model: {model_type}")
        
        try:
            from .models_extended import create_model
            
            config = self._load_model_config(model_type)
            
            model = create_model(model_type, config)
            if model.load():
                self._logger.info(f"Loaded: {model_type}")
                return model
            else:
                self._logger.error(f"Failed to load: {model_type}")
                return None
                
        except Exception as e:
            self._logger.exception(f"Error loading {model_type}: {e}")
            return None
    
    def _load_model_config(self, model_type: str) -> Dict[str, Any]:
        """Load configuration for a model type."""
        import yaml
        
        config = {}
        
        # Try extended config first
        config_paths = [
            Path(__file__).parent.parent.parent / "configs" / "model_config_extended.yaml",
            Path(__file__).parent.parent.parent / "configs" / "model_config.yaml",
        ]
        
        for config_path in config_paths:
            if config_path.exists():
                with open(config_path) as f:
                    full_config = yaml.safe_load(f)
                
                model_section = full_config.get(model_type, {})
                common = full_config.get("common", {})
                
                config = {**common, **model_section}
                break
        
        return config
    
    def generate(
        self,
        image: Union[str, Path],
        prompt: str,
        negative_prompt: Optional[str] = None,
        duration: float = 3.0,
        fps: int = 8,
        width: int = 512,
        height: int = 512,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate video with automatic fallback.
        
        Args:
            image: Input image path
            prompt: Animation prompt
            negative_prompt: Negative prompt
            duration: Video duration in seconds
            fps: FPS
            width: Output width  
            height: Output height
            output_path: Output path
            
        Returns:
            Dictionary with result
        """
        self._attempts = []
        start_time = time.time()
        
        self._logger.info(f"Starting generation with {len(self._model_order)} model(s)")
        
        # Try each model
        for model_type in self._model_order:
            attempt_start = time.time()
            self._logger.info(f"Trying: {model_type}")
            
            model = self.load_model(model_type)
            
            if model is None:
                self._attempts.append(ModelAttempt(
                    model_type=model_type,
                    success=False,
                    error="Failed to load",
                ))
                continue
            
            try:
                # Calculate frames from duration
                num_frames = int(duration * fps)
                
                result = model.generate(
                    image=image,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_frames=num_frames,
                    output_path=output_path,
                )
                
                generation_time = time.time() - attempt_start
                
                if result.success:
                    self._logger.info(f"Success with {model_type} in {generation_time:.1f}s")
                    
                    return {
                        "success": True,
                        "video_path": result.video_path,
                        "model_type": model_type,
                        "generation_time": generation_time,
                        "total_time": time.time() - start_time,
                        "attempts": [
                            {"model": a.model_type, "success": a.success, 
                             "time": a.generation_time, "error": a.error}
                            for a in self._attempts
                        ],
                    }
                else:
                    self._logger.warning(f"{model_type} failed: {result.error}")
                    self._attempts.append(ModelAttempt(
                        model_type=model_type,
                        success=False,
                        error=result.error,
                        generation_time=generation_time,
                    ))
                    
                    # Clean up and try next
                    model.unload()
                    continue
                    
            except Exception as e:
                self._logger.exception(f"Error with {model_type}: {e}")
                self._attempts.append(ModelAttempt(
                    model_type=model_type,
                    success=False,
                    error=str(e),
                    generation_time=time.time() - attempt_start,
                ))
                
                try:
                    model.unload()
                except:
                    pass
                continue
        
        # All failed
        self._logger.error("All models failed")
        
        return {
            "success": False,
            "error": "All models failed",
            "model_type": None,
            "attempts": [
                {"model": a.model_type, "success": a.success, 
                 "time": a.generation_time, "error": a.error}
                for a in self._attempts
            ],
            "total_time": time.time() - start_time,
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get manager status."""
        gpu_name = None
        vram_gb = 0
        
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / (1024**3)
            gpu_name = props.name
        
        return {
            "primary": self.primary,
            "fallback": self.fallback,
            "model_order": self._model_order,
            "gpu_available": torch.cuda.is_available(),
            "gpu_name": gpu_name,
            "vram_gb": round(vram_gb, 1),
            "available_models": {
                m: MODEL_INFO[m].display_name 
                for m in self._model_order
            },
        }
    
    def get_model_info(self, model_type: str) -> Optional[ModelInfo]:
        """Get information about a model."""
        return MODEL_INFO.get(model_type)


# =============================================================================
# Convenience Functions
# =============================================================================

_default_manager: Optional[ModelManager] = None


def get_manager() -> ModelManager:
    """Get default model manager."""
    global _default_manager
    
    if _default_manager is None:
        _default_manager = ModelManager()
    
    return _default_manager


def generate_video(
    image: Union[str, Path],
    prompt: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to generate video.
    
    Usage:
        result = generate_video("input.jpg", "gentle wave")
        if result["success"]:
            print(f"Video: {result['video_path']}")
    """
    manager = get_manager()
    return manager.generate(image, prompt, **kwargs)


def list_models() -> List[Dict[str, Any]]:
    """List all available models with their information."""
    manager = get_manager()
    status = manager.get_status()
    
    models = []
    for model_type in MODEL_INFO:
        info = MODEL_INFO[model_type]
        models.append({
            "type": model_type,
            "name": info.display_name,
            "license": info.license,
            "min_vram_gb": info.min_vram_gb,
            "quality": info.quality,
            "speed": info.speed,
            "available": model_type in status["model_order"],
        })
    
    return models


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Picture-Aliver Model Manager")
    parser.add_argument("--image", help="Input image path")
    parser.add_argument("--prompt", help="Animation prompt")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--status", action="store_true", help="Show status")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s"
    )
    
    if args.list:
        print("\n=== Available Models ===")
        for model in list_models():
            status = "[x]" if model["available"] else "[ ]"
            print(f"{status} {model['name']} ({model['type']})")
            print(f"    License: {model['license']}, VRAM: {model['min_vram_gb']}GB")
            print(f"    Quality: {model['quality']}, Speed: {model['speed']}")
    
    elif args.status:
        manager = get_manager()
        print(f"\n=== Status ===")
        for key, value in manager.get_status().items():
            print(f"{key}: {value}")
    
    elif args.image and args.prompt:
        manager = get_manager()
        print(f"\nGenerating: {args.image}")
        result = manager.generate(args.image, args.prompt)
        
        print(f"\nResult:")
        print(f"  Success: {result['success']}")
        
        if result["success"]:
            print(f"  Video: {result['video_path']}")
            print(f"  Model: {result['model_type']}")
            print(f"  Time: {result['generation_time']:.1f}s")
        else:
            print(f"  Error: {result.get('error')}")
        
        print(f"\nAttempts:")
        for attempt in result.get("attempts", []):
            status = "OK" if attempt["success"] else "FAIL"
            print(f"  [{status}] {attempt['model']}: {attempt.get('time', 0):.1f}s")