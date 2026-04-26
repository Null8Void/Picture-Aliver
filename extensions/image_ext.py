"""
Image Extension Pipeline

Extended image generation that can use new models or fallback to existing.

Behavior:
- If new model selected → use extension pipeline
- Else → call existing image generation function

Returns same format as existing system for compatibility.
"""

from __future__ import annotations

import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union, Tuple
from dataclasses import dataclass
from PIL import Image

logger = logging.getLogger("picture_aliver.extensions.image")


# =============================================================================
# Result Types (matching existing system)
# =============================================================================

@dataclass
class ImageResult:
    """Result of image generation - matches existing format."""
    success: bool
    image: Optional[Image.Image] = None
    image_path: Optional[str] = None
    error: Optional[str] = None
    model_used: str = "default"
    generation_time: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# =============================================================================
# Main Image Generation Function
# =============================================================================

def generate_image_extended(
    prompt: str,
    # Generation params
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    steps: int = 30,
    cfg: float = 7.5,
    seed: int = -1,
    # Model selection
    model_id: Optional[str] = None,
    use_router: bool = True,
    # Output
    save_path: Optional[str] = None,
    # Existing pipeline fallback
    existing_generator=None,
    existing_params: Optional[Dict] = None,
) -> ImageResult:
    """
    Generate image with optional model routing.
    
    Args:
        prompt: Text prompt
        negative_prompt: Negative prompt
        width: Image width
        height: Image height  
        steps: Number of inference steps
        cfg: Guidance scale
        seed: Random seed (-1 for random)
        model_id: Specific model ID (optional)
        use_router: Use keyword router to auto-select model
        save_path: Path to save image
        existing_generator: Fallback to existing generator
        existing_params: Params for existing generator
        
    Returns:
        ImageResult with same format as existing system
    """
    start_time = time.time()
    
    # Step 1: Determine model
    selected_model = None
    
    if model_id:
        # Explicit model requested
        selected_model = model_id
        logger.info(f"Using explicitly requested model: {model_id}")
        
    elif use_router:
        # Use router to detect
        from .router import select_model
        selected_model = select_model(prompt)
        
        if selected_model:
            logger.info(f"Router selected model: {selected_model}")
        else:
            logger.info("Router returned None - using default")
    
    # Step 2: Generate with selected model
    if selected_model:
        try:
            result = _generate_with_model(
                model_id=selected_model,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                cfg=cfg,
                seed=seed,
            )
            
            if result.success:
                # Save if requested
                if save_path:
                    result.image.save(save_path)
                    result.image_path = save_path
                
                result.generation_time = time.time() - start_time
                return result
            
            # Model failed - try fallback
            logger.warning(f"Model {selected_model} failed: {result.error}")
            
        except Exception as e:
            logger.warning(f"Model generation error: {e}")
    
    # Step 3: Fallback to existing pipeline
    logger.info("Falling back to existing pipeline")
    
    return _generate_with_existing(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        steps=steps,
        cfg=cfg,
        seed=seed,
        generator=existing_generator,
        params=existing_params,
        save_path=save_path,
    )


def _generate_with_model(
    model_id: str,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    seed: int,
) -> ImageResult:
    """Generate using new model extension."""
    try:
        # Try to use model loader
        from .models import SafeModelLoader
        
        loader = SafeModelLoader()
        
        result = loader.generate_image(
            prompt=prompt,
            model_id=model_id,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=cfg,
            seed=seed,
        )
        
        return ImageResult(
            success=result.success,
            image=result.image if result.success else None,
            error=result.error,
            model_used=model_id,
        )
        
    except Exception as e:
        return ImageResult(
            success=False,
            error=str(e),
            model_used=model_id,
        )


def _generate_with_existing(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    seed: int,
    generator,
    params: Optional[Dict],
    save_path: Optional[str],
) -> ImageResult:
    """Generate using existing pipeline."""
    try:
        # Try to import existing generator
        if generator is None:
            project_root = Path(__file__).parent.parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            
            from src.picture_aliver.text_to_image import TextToImageGenerator
            generator = TextToImageGenerator()
        
        # Use existing params or defaults
        gen_params = params or {}
        
        # Call existing generator
        result = generator.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_steps=steps,
            guidance_scale=cfg,
            seed=seed,
            **gen_params
        )
        
        # Handle result
        if result and hasattr(result, "image"):
            image = result.image
            
            # Save if path provided
            if save_path:
                image.save(save_path)
            
            return ImageResult(
                success=True,
                image=image,
                image_path=save_path,
                model_used="legacy",
            )
        
        return ImageResult(
            success=False,
            error="Existing generator returned no image",
            model_used="legacy",
        )
        
    except Exception as e:
        return ImageResult(
            success=False,
            error=f"Existing pipeline error: {e}",
            model_used="legacy",
        )


# =============================================================================
# Batch Generation
# =============================================================================

def generate_batch(
    prompts: list,
    **kwargs
) -> list[ImageResult]:
    """Generate multiple images from prompts."""
    results = []
    
    for prompt in prompts:
        result = generate_image_extended(prompt, **kwargs)
        results.append(result)
    
    return results


# =============================================================================
# Convenience Wrapper
# =============================================================================

def generate(
    prompt: str,
    **kwargs
) -> ImageResult:
    """Convenience wrapper matching existing API."""
    return generate_image_extended(prompt, **kwargs)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    
    print("=== Image Extension Test ===\n")
    
    # Test prompts
    prompts = [
        "A beautiful anime girl",
        "Photorealistic portrait",
        "Fantasy castle",
    ]
    
    for prompt in prompts:
        result = generate_image_extended(
            prompt,
            width=512,
            height=512,
            steps=20,
        )
        
        print(f"Prompt: {prompt}")
        print(f"  Success: {result.success}")
        print(f"  Model: {result.model_used}")
        print(f"  Error: {result.error}")
        print()