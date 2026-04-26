"""
Safe Model Loader with Fallback

Provides safe model loading with automatic fallback to existing system.
Non-destructive - if models fail to load, falls back gracefully.

Usage:
    from extensions.models.loader import SafeModelLoader
    
    loader = SafeModelLoader()
    result = loader.generate_image(prompt="...", model_id="dreamshaper")
"""

from __future__ import annotations

import os
import sys
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("picture_aliver.extensions.models.loader")

import torch
from PIL import Image


# =============================================================================
# Result Types
# =============================================================================

@dataclass
class ImageGenerationResult:
    """Result of image generation."""
    success: bool
    image: Optional[Image.Image] = None
    image_path: Optional[str] = None
    error: Optional[str] = None
    model_used: str = ""
    generation_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MotionGenerationResult:
    """Result of motion/video generation."""
    success: bool
    video_path: Optional[str] = None
    error: Optional[str] = None
    model_used: str = ""
    generation_time: float = 0.0
    frames_generated: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# SAFE MODEL LOADER
# =============================================================================

class SafeModelLoader:
    """
    Safe model loader with fallback to existing pipeline.
    
    If new multi-model extension fails, falls back to legacy pipeline.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize loader with optional config."""
        self._registry = None
        self._config = self._load_config(config_path)
        self._fallback_pipeline = None
        
        # Determine device
        self._device = self._config.get("default_device", "auto")
        if self._device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Determine dtype
        dtype_str = self._config.get("default_dtype", "float16")
        self._torch_dtype = getattr(torch, dtype_str, torch.float16)
        
        self._logger = logging.getLogger("picture_aliver.extensions.models.loader")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration."""
        import yaml
        
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        
        config_path = Path(config_path)
        
        if not config_path.exists():
            return {}
        
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    
    # =========================================================================
    # Image Generation
    # =========================================================================
    
    def generate_image(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: int = -1,
        **kwargs
    ) -> ImageGenerationResult:
        """
        Generate an image with fallback.
        
        Args:
            prompt: Text prompt
            model_id: Specific model to use (optional)
            negative_prompt: Negative prompt
            width: Image width
            height: Image height
            num_inference_steps: Number of steps
            guidance_scale: CFG scale
            seed: Random seed (-1 for random)
            
        Returns:
            ImageGenerationResult
        """
        model_id = model_id or self._config.get("default_model", "sdxl_base")
        start_time = time.time()
        
        self._logger.info(f"Generating with model: {model_id}")
        
        # Try new model loader first
        try:
            result = self._generate_with_extension(
                prompt=prompt,
                model_id=model_id,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                **kwargs
            )
            
            if result.success:
                self._logger.info(f"Generated with {model_id} in {result.generation_time:.1f}s")
                return result
            
            self._logger.warning(f"Model {model_id} failed: {result.error}")
            
        except Exception as e:
            self._logger.warning(f"Extension generation failed: {e}")
        
        # Fall back to legacy pipeline
        self._logger.info("Falling back to legacy pipeline")
        
        try:
            result = self._generate_with_legacy(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
            )
            
            if result.success:
                self._logger.info(f"Generated with legacy in {result.generation_time:.1f}s")
                return result
            
        except Exception as e:
            self._logger.error(f"Legacy generation also failed: {e}")
        
        return ImageGenerationResult(
            success=False,
            error="All generation methods failed",
            model_used=model_id,
        )
    
    def _generate_with_extension(
        self,
        prompt: str,
        model_id: str,
        **kwargs
    ) -> ImageGenerationResult:
        """Generate using the new model extension."""
        # Get registry
        if self._registry is None:
            from .registry import get_registry
            self._registry = get_registry()
        
        # Check if model is motion
        model_info = self._registry.get(model_id)
        
        if model_info and model_info.is_motion:
            # Use motion generation
            return self._generate_motion(
                prompt=model_id,
                **kwargs
            )
        
        # Get or load model
        model = self._registry.load(
            model_id,
            device=self._device,
            torch_dtype=self._torch_dtype,
        )
        
        if model is None:
            return ImageGenerationResult(
                success=False,
                error=f"Failed to load model: {model_id}",
                model_used=model_id,
            )
        
        # Generate
        generator = None
        if kwargs.get("seed", -1) >= 0:
            generator = torch.Generator(device=self._device).manual_seed(kwargs["seed"])
        
        result = model(
            prompt=prompt,
            negative_prompt=kwargs.get("negative_prompt", ""),
            width=kwargs.get("width", 1024),
            height=kwargs.get("height", 1024),
            num_inference_steps=kwargs.get("num_inference_steps", 30),
            guidance_scale=kwargs.get("guidance_scale", 7.5),
            generator=generator,
        )
        
        # Extract image
        image = result.images[0] if hasattr(result, "images") else result[0]
        
        return ImageGenerationResult(
            success=True,
            image=image,
            model_used=model_id,
            generation_time=0.0,  # Will be calculated by caller
        )
    
    def _generate_with_legacy(
        self,
        prompt: str,
        **kwargs
    ) -> ImageGenerationResult:
        """Generate using legacy pipeline (SAFE fallback)."""
        try:
            # Import from picture_aliver
            project_root = Path(__file__).parent.parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            
            from src.picture_aliver.text_to_image import TextToImageGenerator
            
            # Use default size
            width = kwargs.get("width", 512)
            height = kwargs.get("height", 512)
            
            # Create generator with default config
            generator = TextToImageGenerator()
            
            # Generate
            result = generator.generate(
                prompt=prompt,
                width=width,
                height=height,
                num_steps=kwargs.get("num_inference_steps", 25),
                guidance_scale=kwargs.get("guidance_scale", 7.5),
            )
            
            if result and hasattr(result, "image"):
                return ImageGenerationResult(
                    success=True,
                    image=result.image,
                    model_used="legacy",
                )
            
        except Exception as e:
            self._logger.error(f"Legacy generation error: {e}")
        
        return ImageGenerationResult(
            success=False,
            error="Legacy generation failed",
        )
    
    # =========================================================================
    # Motion Generation
    # =========================================================================
    
    def generate_motion(
        self,
        image: Union[str, Image.Image],
        prompt: str,
        model_id: Optional[str] = None,
        duration: float = 2.0,
        fps: int = 8,
        **kwargs
    ) -> MotionGenerationResult:
        """
        Generate motion/video from image.
        
        Args:
            image: Input image
            prompt: Motion prompt  
            model_id: Motion model to use
            duration: Video duration in seconds
            fps: Frames per second
            
        Returns:
            MotionGenerationResult
        """
        model_id = model_id or self._config.get("default_motion_model", "svd")
        start_time = time.time()
        
        self._logger.info(f"Generating motion with: {model_id}")
        
        try:
            if self._registry is None:
                from .registry import get_registry
                self._registry = get_registry()
            
            model = self._registry.load(
                model_id,
                device=self._device,
                torch_dtype=self._torch_dtype,
            )
            
            if model is None:
                return MotionGenerationResult(
                    success=False,
                    error=f"Failed to load motion model: {model_id}",
                )
            
            # Load input image
            if isinstance(image, (str, Path)):
                image = Image.open(image)
            
            # Calculate frames
            num_frames = int(duration * fps)
            
            # Generate
            result = model(
                image=image,
                prompt=prompt,
                num_frames=num_frames,
                **kwargs
            )
            
            # Save video
            output_path = self._save_motion(result, model_id)
            
            return MotionGenerationResult(
                success=True,
                video_path=output_path,
                model_used=model_id,
                generation_time=time.time() - start_time,
                frames_generated=num_frames,
            )
            
        except Exception as e:
            self._logger.error(f"Motion generation failed: {e}")
            return MotionGenerationResult(
                success=False,
                error=str(e),
                model_used=model_id,
            )
    
    def _save_motion(self, frames, model_id: str) -> str:
        """Save motion frames as video."""
        from pathlib import Path
        from diffusers.utils import export_to_video
        
        output_dir = Path(__file__).parent.parent.parent / "outputs"
        output_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"motion_{model_id}_{timestamp}.mp4"
        
        export_to_video(frames[0], str(output_path), fps=8)
        
        return str(output_path)
    
    # =========================================================================
    # Utility
    # =========================================================================
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models."""
        if self._registry is None:
            from .registry import get_registry
            self._registry = get_registry()
        
        return [
            {
                "id": m.id,
                "name": m.name,
                "base": m.base_model,
                "category": m.category.value,
                "license": m.license.value,
                "min_vram_gb": m.min_vram_gb,
                "is_motion": m.is_motion,
            }
            for m in self._registry.list_models()
        ]
    
    def get_vram_info(self) -> Dict[str, Any]:
        """Get VRAM information."""
        info = {
            "cuda_available": torch.cuda.is_available(),
            "device_used": self._device,
        }
        
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["total_vram_gb"] = props.total_memory / (1024**3)
            info["allocated_vram_gb"] = torch.cuda.memory_allocated(0) / (1024**3)
            info["free_vram_gb"] = info["total_vram_gb"] - info["allocated_vram_gb"]
            info["gpu_name"] = props.name
        
        return info


# =============================================================================
# Convenience Functions
# =============================================================================

_loader_instance: Optional[SafeModelLoader] = None


def get_loader() -> SafeModelLoader:
    """Get singleton loader instance."""
    global _loader_instance
    
    if _loader_instance is None:
        _loader_instance = SafeModelLoader()
    
    return _loader_instance


def generate_image(prompt: str, **kwargs) -> ImageGenerationResult:
    """Convenience function for image generation."""
    return get_loader().generate_image(prompt, **kwargs)


def generate_motion(image: Union[str, Image.Image], prompt: str, **kwargs) -> MotionGenerationResult:
    """Convenience function for motion generation."""
    return get_loader().generate_motion(image, prompt, **kwargs)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Safe Model Loader")
    parser.add_argument("--models", action="store_true", help="List models")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--generate", help="Generate image with prompt")
    parser.add_argument("--model", default="sdxl_base", help="Model to use")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    loader = get_loader()
    
    if args.models:
        print("\n=== Available Models ===")
        for model in loader.get_available_models():
            marker = "*" if model["is_motion"] else " "
            print(f"{marker} {model['id']}: {model['name']} ({model['base']})")
    
    elif args.status:
        print("\n=== VRAM Status ===")
        info = loader.get_vram_info()
        for k, v in info.items():
            print(f"  {k}: {v}")
    
    elif args.generate:
        result = loader.generate_image(args.generate, args.model)
        print(f"\nResult: {result.success}")
        if result.success:
            print(f"  Model: {result.model_used}")
            print(f"  Saved to: {result.image_path}")
        else:
            print(f"  Error: {result.error}")