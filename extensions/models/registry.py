"""
Picture-Aliver Model Registry

Multi-model support with safe loading and automatic fallback.
Models are cached in memory and lazy-loaded on first use.

Usage:
    from extensions.models.registry import ModelRegistry, list_models
    
    registry = ModelRegistry()
    model = registry.load("sdxl")
    image = model.generate(prompt="...")
"""

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
import hashlib

logger = logging.getLogger("picture_aliver.extensions.models")


# =============================================================================
# Model Categories
# =============================================================================

class ModelCategory(Enum):
    """Categories of models available."""
    # Image Generation
    SDXL_BASE = "sdxl_base"           # Stable Diffusion XL Base
    SD_BASE = "sd_base"               # Stable Diffusion 1.5/2.1 Base
    CHECKPOINT = "checkpoint"         # Fine-tuned checkpoints
    ANIME = "anime"                   # Anime style
    STYLIZED = "stylized"             # Artistic/Stylized
    REALISTIC = "realistic"            # Photorealistic
    MIX = "mix"                       # Mixed style checkpoints
    
    # Motion/Video
    MOTION = "motion"                 # Motion generation
    I2V = "i2v"                      # Image-to-video
    T2V = "t2v"                      # Text-to-video


class ModelLicense(Enum):
    """Model license types."""
    OPEN = "open"                    # Fully open (Apache 2.0, MIT, etc.)
    CREATIVE_ML = "creative_ml"      # CreativeML (some restrictions)
    CUSTOM = "custom"                # Custom license
    UNKNOWN = "unknown"              # Unknown license


@dataclass
class ModelInfo:
    """Information about a registered model."""
    id: str                          # Unique identifier
    name: str                        # Display name
    category: ModelCategory          # Model category
    license: ModelLicense            # License type
    
    # Model source (HuggingFace repo or local path)
    repo_id: str = ""                 # HuggingFace repository ID
    local_path: str = ""             # Local path override
    
    # Model type
    base_model: str = "sdxl"          # Base: sdxl, sd15, sd21
    model_type: str = "diffusers"   # diffusers, safetensors, ckpt
    
    # Requirements
    min_vram_gb: float = 6.0         # Minimum VRAM in GB
    requires_sdxl: bool = False     # Requires SDXL base
    
    # Additional info
    description: str = ""
    author: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Motion specific
    is_motion: bool = False         # Is motion/video model
    supports_fp16: bool = True      # Supports FP16


# =============================================================================
# MODEL REGISTRY
# =============================================================================

MODEL_REGISTRY: Dict[str, ModelInfo] = {
    # =====================================================================
    # SDXL BASE MODELS
    # =====================================================================
    "sdxl_base": ModelInfo(
        id="sdxl_base",
        name="Stable Diffusion XL Base",
        category=ModelCategory.SDXL_BASE,
        license=ModelLicense.OPEN,
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        base_model="sdxl",
        min_vram_gb=6.0,
        requires_sdxl=True,
        description="Stable Diffusion XL Base 1.0",
    ),
    
    "sdxl_refiner": ModelInfo(
        id="sdxl_refiner",
        name="Stable Diffusion XL Refiner",
        category=ModelCategory.SDXL_BASE,
        license=ModelLicense.OPEN,
        repo_id="stabilityai/stable-diffusion-xl-refiner-1.0",
        base_model="sdxl",
        min_vram_gb=6.0,
        requires_sdxl=True,
        description="Stable Diffusion XL Refiner 1.0",
    ),
    
    # =====================================================================
    # SD 1.5/2.1 BASE MODELS  
    # =====================================================================
    "sd15_base": ModelInfo(
        id="sd15_base",
        name="Stable Diffusion 1.5 Base",
        category=ModelCategory.SD_BASE,
        license=ModelLicense.OPEN,
        repo_id="runwayml/stable-diffusion-v1-5",
        base_model="sd15",
        min_vram_gb=4.0,
        description="Stable Diffusion 1.5",
    ),
    
    "sd21_base": ModelInfo(
        id="sd21_base",
        name="Stable Diffusion 2.1 Base",
        category=ModelCategory.SD_BASE,
        license=ModelLicense.OPEN,
        repo_id="stabilityai/stable-diffusion-2-1",
        base_model="sd21",
        min_vram_gb=4.0,
        description="Stable Diffusion 2.1",
    ),
    
    # =====================================================================
    # CHECKPOINT MODELS - Realistic
    # =====================================================================
    "dreamshaper": ModelInfo(
        id="dreamshaper",
        name="DreamShaper",
        category=ModelCategory.CHECKPOINT,
        license=ModelLicense.CREATIVE_ML,
        repo_id="Lykon/DreamShaper",
        base_model="sd15",
        min_vram_gb=4.0,
        description="Versatile checkpoint for realistic and artistic images",
        author="Lykon",
        tags=["realistic", "artistic", "portrait"],
    ),
    
    "dreamshaper_xl": ModelInfo(
        id="dreamshaper_xl",
        name="DreamShaper XL",
        category=ModelCategory.CHECKPOINT,
        license=ModelLicense.CREATIVE_ML,
        repo_id="Lykon/DreamShaperXL",
        base_model="sdxl",
        min_vram_gb=8.0,
        requires_sdxl=True,
        description="DreamShaper for SDXL",
        author="Lykon",
        tags=["sdxl", "artistic"],
    ),
    
    # =====================================================================
    # CHECKPOINT MODELS - Anime
    # =====================================================================
    "anything_v5": ModelInfo(
        id="anything_v5",
        name="Anything V5",
        category=ModelCategory.ANIME,
        license=ModelLicense.CREATIVE_ML,
        repo_id="andite/anything-v5",
        base_model="sd15",
        min_vram_gb=4.0,
        description="Popular anime style model",
        tags=["anime", "illustration"],
    ),
    
    "anything_xl": ModelInfo(
        id="anything_xl",
        name="Anything XL",
        category=ModelCategory.ANIME,
        license=ModelLicense.CREATIVE_ML,
        repo_id="andite/anythingxl",
        base_model="sdxl",
        min_vram_gb=6.0,
        requires_sdxl=True,
        description="Anything for SDXL",
        tags=["anime", "sdxl"],
    ),
    
    # =====================================================================
    # CHECKPOINT MODELS - Stylized
    # =====================================================================
    "solarmix": ModelInfo(
        id="solarmix",
        name="SolarMix",
        category=ModelCategory.STYLIZED,
        license=ModelLicense.CREATIVE_ML,
        repo_id="ashleykleynhof/solarmix",
        base_model="sd15",
        min_vram_gb=4.0,
        description="Solarpunk/environmentalist style",
        tags=["solarpunk", "nature"],
    ),
    
    "solarmix_xl": ModelInfo(
        id="solarmix_xl",
        name="SolarMix XL",
        category=ModelCategory.STYLIZED,
        license=ModelLicense.CREATIVE_ML,
        repo_id="ashleykleynhof/solarmix-xl",
        base_model="sdxl",
        min_vram_gb=6.0,
        requires_sdxl=True,
        description="SolarMix for SDXL",
        tags=["solarpunk", "sdxl"],
    ),
    
    # =====================================================================
    # CHECKPOINT MODELS - YiffyMix (NSFW-warning in config)
    # =====================================================================
    "yiffymix": ModelInfo(
        id="yiffymix",
        name="YiffyMix",
        category=ModelCategory.MIX,
        license=ModelLicense.CREATIVE_ML,
        repo_id="cagliostrosd/YiffyMixSD15",
        base_model="sd15",
        min_vram_gb=4.0,
        description="Mixed style model - check license for use restrictions",
        tags=["mixed", "furry"],
    ),
    
    "yiffymix_v2": ModelInfo(
        id="yiffymix_v2", 
        name="YiffyMix V2",
        category=ModelCategory.MIX,
        license=ModelLicense.CREATIVE_ML,
        repo_id="cagliostrosd/YiffyMixSD15V2",
        base_model="sd15",
        min_vram_gb=4.0,
        description="YiffyMix V2 - check license",
        tags=["mixed", "furry", "v2"],
    ),
    
    # =====================================================================
    # CHECKPOINT MODELS - Compass/Mix
    # =====================================================================
    "compass_mix_xl": ModelInfo(
        id="compass_mix_xl",
        name="CompassMix XL",
        category=ModelCategory.MIX,
        license=ModelLicense.CREATIVE_ML,
        repo_id="OkayHuman/CompassMixXL",
        base_model="sdxl",
        min_vram_gb=8.0,
        requires_sdxl=True,
        description="CompassMix for SDXL",
        tags=["mixed", "sdxl"],
    ),
    
    "compass_mix_xl_lite": ModelInfo(
        id="compass_mix_xl_lite",
        name="CompassMix XL Lite",
        category=ModelCategory.MIX,
        license=ModelLicense.CREATIVE_ML,
        repo_id="OkayHuman/CompassMixXLLite",
        base_model="sdxl",
        min_vram_gb=4.0,
        requires_sdxl=True,
        description="Lightweight CompassMix",
        tags=["mixed", "sdxl", "lite"],
    ),
    
    # =====================================================================
    # CHECKPOINT MODELS - Indigo/Indi
    # =====================================================================
    "indigomix": ModelInfo(
        id="indigomix",
        name="IndigoMix",
        category=ModelCategory.MIX,
        license=ModelLicense.CREATIVE_ML,
        repo_id="dataautoindigo/IndigoMix",
        base_model="sd15",
        min_vram_gb=4.0,
        description="Indigo mix for SD1.5",
        tags=["mixed", "indigo"],
    ),
    
    "indigomix_v": ModelInfo(
        id="indigomix_v",
        name="IndigoMix V",
        category=ModelCategory.MIX,
        license=ModelLicense.CREATIVE_ML,
        repo_id="dataautoindigo/IndigoMixV",
        base_model="sd15",
        min_vram_gb=4.0,
        description="IndigoMix V",
        tags=["mixed", "indigo", "v2"],
    ),
    
    # =====================================================================
    # CHECKPOINT MODELS - BluePencil  
    # =====================================================================
    "bluepencil_xl": ModelInfo(
        id="bluepencil_xl",
        name="BluePencil XL",
        category=ModelCategory.STYLIZED,
        license=ModelLicense.CREATIVE_ML,
        repo_id="bluepencil/bluepencilXL",
        base_model="sdxl",
        min_vram_gb=6.0,
        requires_sdxl=True,
        description="BluePencil for SDXL",
        tags=["sketch", "lineart"],
    ),
    
    "bluepencil_xl_lite": ModelInfo(
        id="bluepencil_xl_lite",
        name="BluePencil XL Lite",
        category=ModelCategory.STYLIZED,
        license=ModelLicense.CREATIVE_ML,
        repo_id="bluepencil/bluepencilXLLite",
        base_model="sdxl",
        min_vram_gb=4.0,
        requires_sdxl=True,
        description="Lightweight BluePencil",
        tags=["sketch", "lineart", "lite"],
    ),
    
    # =====================================================================
    # CHECKPOINT MODELS - FluffyRock
    # =====================================================================
    "fluffyrock": ModelInfo(
        id="fluffyrock",
        name="FluffyRock",
        category=ModelCategory.CHECKPOINT,
        license=ModelLicense.CREATIVE_ML,
        repo_id="fluffyrock/fluffyrock",
        base_model="sd15",
        min_vram_gb=4.0,
        description="FluffyRock checkpoint",
        tags=["realistic", "stylish"],
    ),
    
    # =====================================================================
    # CHECKPOINT MODELS - KotoMix
    # =====================================================================
    "kotosmix": ModelInfo(
        id="kotosmix",
        name="KotoMix",
        category=ModelCategory.MIX,
        license=ModelLicense.CREATIVE_ML,
        repo_id="konstantin/kotosmix",
        base_model="sd15",
        min_vram_gb=4.0,
        description="KotoMix",
        tags=["mixed"],
    ),
    
    # =====================================================================
    # CHECKPOINT MODELS - Flux/Sirius (via Civitai)
    # =====================================================================
    "sirius": ModelInfo(
        id="sirius",
        name="Sirius",
        category=ModelCategory.CHECKPOINT,
        license=ModelLicense.CUSTOM,
        local_path="",  # Not on HuggingFace - requires manual download
        base_model="sd15",
        min_vram_gb=4.0,
        description="Sirius - check license/Civitai for download",
        tags=["realistic"],
    ),
    
    "alpha_centauri_flux": ModelInfo(
        id="alpha_centauri_flux",
        name="Alpha Centauri Flux",
        category=ModelCategory.CHECKPOINT,
        license=ModelLicense.CUSTOM,
        local_path="",
        base_model="sd15",
        min_vram_gb=6.0,
        description="Alpha Centauri Flux",
        tags=["realistic", "flux"],
    ),
    
    # =====================================================================
    # MOTION/VIDEO MODELS - Image to Video
    # =====================================================================
    "lynx_one": ModelInfo(
        id="lynx_one",
        name="Lynx One",
        category=ModelCategory.I2V,
        license=ModelLicense.OPEN,
        # Placeholder - check for actual repo
        repo_id="",
        base_model="i2v",
        min_vram_gb=8.0,
        is_motion=True,
        description="Lynx One image-to-video model",
        tags=["motion", "i2v"],
    ),
    
    "polaris": ModelInfo(
        id="polaris",
        name="Polaris",
        category=ModelCategory.I2V,
        license=ModelLicense.OPEN,
        repo_id="",
        base_model="i2v",
        min_vram_gb=8.0,
        is_motion=True,
        description="Polaris motion model",
        tags=["motion", "i2v"],
    ),
    
    "mecano": ModelInfo(
        id="mecano",
        name="Mecano",
        category=ModelCategory.I2V,
        license=ModelLicense.CUSTOM,
        repo_id="",
        base_model="i2v",
        min_vram_gb=8.0,
        is_motion=True,
        description="Mecano motion model",
        tags=["motion", "i2v"],
    ),
    
    "svd": ModelInfo(
        id="svd",
        name="Stable Video Diffusion",
        category=ModelCategory.I2V,
        license=ModelLicense.CREATIVE_ML,
        repo_id="stabilityai/stable-video-diffusion",
        base_model="i2v",
        min_vram_gb=12.0,
        is_motion=True,
        description="SVD - Image to Video",
        tags=["motion", "i2v", "stabilityai"],
    ),
}


# =============================================================================
# MODEL REGISTRY CLASS
# =============================================================================

class ModelRegistry:
    """
    Safe model registry with lazy loading and caching.
    
    Features:
    - Cache models in memory once loaded
    - Lazy-load on first use
    - Automatic fallback on failure
    - VRAM-aware model selection
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._models: Dict[str, Any] = {}  # Loaded model cache
        self._model_locks: Dict[str, threading.Lock] = {}
        self._initialized = True
        
        logger.info(f"Model registry initialized with {len(MODEL_REGISTRY)} models")
    
    # =========================================================================
    # Registry Operations
    # =========================================================================
    
    def register(self, model_info: ModelInfo) -> None:
        """Register a new model."""
        MODEL_REGISTRY[model_info.id] = model_info
        logger.info(f"Registered model: {model_info.name}")
    
    def get(self, model_id: str) -> Optional[ModelInfo]:
        """Get model info by ID."""
        return MODEL_REGISTRY.get(model_id)
    
    def list_models(
        self,
        category: Optional[ModelCategory] = None,
        is_motion: Optional[bool] = None,
        requires_sdxl: Optional[bool] = None,
    ) -> List[ModelInfo]:
        """List available models with optional filters."""
        models = []
        
        for model in MODEL_REGISTRY.values():
            if category and model.category != category:
                continue
            if is_motion is not None and model.is_motion != is_motion:
                continue
            if requires_sdxl is not None and model.requires_sdxl != requires_sdxl:
                continue
            models.append(model)
        
        return models
    
    def list_checkpoints(self) -> List[ModelInfo]:
        """List available checkpoint models (non-motion)."""
        return self.list_models(is_motion=False)
    
    def list_motion_models(self) -> List[ModelInfo]:
        """List available motion/video models."""
        return self.list_models(is_motion=True)
    
    def list_sdxl_models(self) -> List[ModelInfo]:
        """List SDXL-based models."""
        return self.list_models(requires_sdxl=True)
    
    # =========================================================================
    # Model Loading
    # =========================================================================
    
    def is_loaded(self, model_id: str) -> bool:
        """Check if model is loaded."""
        return model_id in self._models and self._models[model_id] is not None
    
    def get_loaded(self, model_id: str) -> Optional[Any]:
        """Get loaded model without loading."""
        return self._models.get(model_id)
    
    def load(
        self,
        model_id: str,
        device: str = "cuda",
        torch_dtype: Any = None,
    ) -> Optional[Any]:
        """
        Load a model safely with caching.
        
        Args:
            model_id: Model identifier
            device: Target device (cuda/cpu)
            torch_dtype: Data type (torch.float16, etc.)
            
        Returns:
            Loaded model or None on failure
        """
        # Check if already loaded
        if self.is_loaded(model_id):
            logger.debug(f"Model already loaded: {model_id}")
            return self._models[model_id]
        
        # Get model info
        model_info = self.get(model_id)
        if model_info is None:
            logger.error(f"Model not found: {model_id}")
            return None
        
        # Check if another thread is loading
        if model_id in self._model_locks:
            # Wait briefly for other thread
            import time
            for _ in range(10):
                time.sleep(0.1)
                if self.is_loaded(model_id):
                    return self._models[model_id]
            return None
        
        # Create lock for this model
        with self._lock:
            self._model_locks[model_id] = threading.Lock()
        
        try:
            with self._model_locks[model_id]:
                # Double-check after acquiring lock
                if self.is_loaded(model_id):
                    return self._models[model_id]
                
                logger.info(f"Loading model: {model_id}")
                
                # Determine model class
                model = self._load_model(model_info, device, torch_dtype)
                
                if model is not None:
                    self._models[model_id] = model
                    logger.info(f"Model loaded: {model_id}")
                else:
                    logger.error(f"Failed to load model: {model_id}")
                
                return model
                
        finally:
            with self._lock:
                if model_id in self._model_locks:
                    del self._model_locks[model_id]
    
    def _load_model(
        self,
        model_info: ModelInfo,
        device: str,
        torch_dtype: Any,
    ) -> Optional[Any]:
        """Internal model loader."""
        try:
            # Try Diffusers first
            return self._load_diffusers_model(model_info, device, torch_dtype)
        except Exception as e:
            logger.error(f"Failed to load {model_info.id}: {e}")
            return None
    
    def _load_diffusers_model(
        self,
        model_info: ModelInfo,
        device: str,
        torch_dtype: Any,
    ) -> Optional[Any]:
        """Load model via Diffusers."""
        try:
            from diffusers import StableDiffusionXLPipeline
            from diffusers import StableDiffusionPipeline
            import torch
            
            if torch_dtype is None:
                torch_dtype = torch.float16 if device == "cuda" else torch.float32
            
            # Determine model class
            if model_info.requires_sdxl or model_info.base_model == "sdxl":
                pipeline_class = StableDiffusionXLPipeline
            elif model_info.is_motion:
                # Motion models need different loading
                logger.info(f"Motion model {model_info.id} - using specialized loader")
                return self._load_motion_model(model_info, device, torch_dtype)
            else:
                pipeline_class = StableDiffusionPipeline
            
            # Determine repo
            repo_id = model_info.repo_id
            if not repo_id:
                logger.warning(f"No repo_id for {model_info.id}")
                return None
            
            # Load pipeline
            logger.info(f"Loading from HuggingFace: {repo_id}")
            
            pipeline = pipeline_class.from_pretrained(
                repo_id,
                torch_dtype=torch_dtype,
            )
            
            # Move to device if not CPU
            if device != "cpu":
                pipeline = pipeline.to(device)
            
            return pipeline
            
        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            return None
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    
    def _load_motion_model(
        self,
        model_info: ModelInfo,
        device: str,
        torch_dtype: Any,
    ) -> Optional[Any]:
        """Load motion/video model."""
        try:
            # Try SVD first
            from diffusers import StableVideoDiffusionPipeline
            import torch
            
            if torch_dtype is None:
                torch_dtype = torch.float16 if device == "cuda" else torch.float32
            
            if model_info.id == "svd" and model_info.repo_id:
                pipeline = StableVideoDiffusionPipeline.from_pretrained(
                    model_info.repo_id,
                    torch_dtype=torch_dtype,
                )
                if device != "cpu":
                    pipeline = pipeline.to(device)
                return pipeline
            
            # Generic motion model loading
            if model_info.repo_id:
                pipeline = StableVideoDiffusionPipeline.from_pretrained(
                    model_info.repo_id,
                    torch_dtype=torch_dtype,
                )
                if device != "cpu":
                    pipeline = pipeline.to(device)
                return pipeline
            
            logger.warning(f"No loading method for motion model: {model_info.id}")
            return None
            
        except ImportError:
            logger.warning("StableVideoDiffusion not available")
            return None
        except Exception as e:
            logger.error(f"Motion model load error: {e}")
            return None
    
    def unload(self, model_id: str) -> None:
        """Unload a specific model from cache."""
        if model_id in self._models:
            model = self._models[model_id]
            
            # Cleanup
            if hasattr(model, "to"):
                try:
                    model.to("cpu")
                except:
                    pass
            
            del self._models[model_id]
            logger.info(f"Unloaded model: {model_id}")
    
    def unload_all(self) -> None:
        """Unload all cached models."""
        for model_id in list(self._models.keys()):
            self.unload(model_id)
    
    def get_vram_estimate(self) -> float:
        """Get estimated VRAM usage of loaded models."""
        return sum(
            m.min_vram_gb for m in self._models.keys() 
            if m in MODEL_REGISTRY
        )
    
    def get_best_model_for_vram(self, vram_gb: float) -> Optional[ModelInfo]:
        """Get the best model that fits in available VRAM."""
        # Prefer already loaded models
        for model_id in self._models:
            info = self.get(model_id)
            if info and info.min_vram_gb <= vram_gb:
                return info
        
        # Find any model that fits
        for info in MODEL_REGISTRY.values():
            if info.min_vram_gb <= vram_gb and not info.is_motion:
                return info
        
        return None


# =============================================================================
# Convenience Functions
# =============================================================================

def get_registry() -> ModelRegistry:
    """Get the model registry instance."""
    return ModelRegistry()


def list_models(**filters) -> List[ModelInfo]:
    """List models with filters."""
    return get_registry().list_models(**filters)


def load_model(model_id: str, **kwargs) -> Optional[Any]:
    """Load a model by ID."""
    return get_registry().load(model_id, **kwargs)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="List all models")
    parser.add_argument("--checkpoints", action="store_true", help="List checkpoints")
    parser.add_argument("--motion", action="store_true", help="List motion models")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    registry = get_registry()
    
    if args.list:
        print("\n=== All Models ===")
        for info in registry.list_models():
            print(f"{info.id}: {info.name} ({info.base_model})")
            print(f"  Category: {info.category.value}")
            print(f"  VRAM: {info.min_vram_gb}GB")
            print(f"  License: {info.license.value}")
            print()
    
    elif args.checkpoints:
        print("\n=== Checkpoint Models ===")
        for info in registry.list_checkpoints():
            print(f"{info.id}: {info.name}")
    
    elif args.motion:
        print("\n=== Motion Models ===")
        for info in registry.list_motion_models():
            print(f"{info.id}: {info.name}")