"""
Picture-Aliver Extensions - Models Package

Multi-model support with safe loading and automatic fallback.
Non-destructive - falls back to existing system if models fail to load.

Usage:
    # List available models
    from extensions.models import list_models
    for model in list_models():
        print(f"{model.id}: {model.name}")
    
    # Generate image with specific model
    from extensions.models import SafeModelLoader
    loader = SafeModelLoader()
    result = loader.generate_image(prompt="...", model_id="dreamshaper")
"""

from .registry import (
    ModelRegistry,
    ModelInfo,
    ModelCategory,
    ModelLicense,
    MODEL_REGISTRY,
    get_registry,
    list_models,
    load_model,
)

from .loader import (
    SafeModelLoader,
    ImageGenerationResult,
    MotionGenerationResult,
    get_loader,
    generate_image,
    generate_motion,
)

__all__ = [
    # Registry
    "ModelRegistry",
    "ModelInfo",
    "ModelCategory", 
    "ModelLicense",
    "MODEL_REGISTRY",
    "get_registry",
    "list_models",
    "load_model",
    # Loader
    "SafeModelLoader",
    "ImageGenerationResult", 
    "MotionGenerationResult",
    "get_loader",
    "generate_image",
    "generate_motion",
]