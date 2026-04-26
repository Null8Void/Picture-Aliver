"""
Picture-Aliver Model Loader with Fallback

Non-destructive extension for multi-model support.
Falls back to existing pipeline if models fail to load.
"""

from .models import (
    SafeModelLoader,
    get_loader,
    generate_image,
    generate_motion,
)

__all__ = ["SafeModelLoader", "get_loader", "generate_image", "generate_motion"]