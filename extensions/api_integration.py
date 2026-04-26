"""
Safe API Integration for Video Extension

This module adds the /generate-video endpoint WITHOUT modifying existing code.
Simply include this router in your FastAPI app to enable video generation.

Usage in existing api.py:
    from extensions.api_integration import router as video_router
    app.include_router(video_router)

OR run separately:
    uvicorn extensions.api_integration:app --host 0.0.0.0 --port 8001
"""

from __future__ import annotations

import os
import sys
import time
import uuid
import asyncio
import logging
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass

from fastapi import FastAPI, APIRouter, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import torch

logger = logging.getLogger("picture_aliver.extensions.api")


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class VideoGenerationRequest(BaseModel):
    """Request for video generation."""
    prompt: str
    negative_prompt: str = ""
    width: int = 576
    height: int = 576
    duration: float = 2.0
    frames: int = 24
    fps: int = 12
    steps: int = 25
    cfg: float = 7.0
    seed: int = -1
    motion_model: str = "auto"


class VideoGenerationResponse(BaseModel):
    """Response for video generation."""
    success: bool
    video_path: Optional[str] = None
    error: Optional[str] = None
    model_used: str
    generation_time: float


class ImageGenerationExtendedRequest(BaseModel):
    """Extended request for image generation (optional fields)."""
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 30
    cfg: float = 7.5
    seed: int = -1
    model: Optional[str] = None
    use_extended: bool = False


class ImageGenerationExtendedResponse(BaseModel):
    """Response for image generation."""
    success: bool
    image_path: Optional[str] = None
    error: Optional[str] = None
    model_used: str
    generation_time: float


# =============================================================================
# FASTAPI ROUTER
# =============================================================================

# Main router - include in existing app or run standalone
router = APIRouter(prefix="/generate-video", tags=["Video Extension"])


# =============================================================================
# IMAGE GENERATION (EXTENDED)
# =============================================================================

@router.post("/image-extended", response_model=ImageGenerationExtendedResponse)
async def generate_image_extended(
    background_tasks: BackgroundTasks,
    image: Optional[UploadFile] = File(None, description="Input image (optional for text-to-image)"),
    prompt: str = Form(..., description="Text prompt"),
    negative_prompt: str = Form(default="", description="Negative prompt"),
    width: int = Form(default=1024, ge=256, le=2048, description="Image width"),
    height: int = Form(default=1024, ge=256, le=2048, description="Image height"),
    steps: int = Form(default=30, ge=1, le=150, description="Inference steps"),
    cfg: float = Form(default=7.5, ge=1.0, le=20.0, description="Guidance scale"),
    seed: int = Form(default=-1, description="Random seed (-1 for random)"),
    model: Optional[str] = Form(default=None, description="Model ID (optional)"),
    use_extended: bool = Form(default=False, description="Use extended model pipeline"),
) -> ImageGenerationExtendedResponse:
    """
    Generate image with optional extended model pipeline.
    
    If model is specified or use_extended=True, uses the new model pipeline.
    Otherwise falls back to existing pipeline.
    
    Extra optional fields (do NOT break existing behavior if missing):
        - model: Specific model ID to use
        - use_extended: Whether to use extended pipeline
    """
    start_time = time.time()
    
    try:
        # Use extended pipeline only if requested
        if model or use_extended:
            from extensions.image_ext import generate_image_extended as gen_extended
            
            result = gen_extended(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                cfg=cfg,
                seed=seed,
                model_id=model,
            )
            
            return ImageGenerationExtendedResponse(
                success=result.success,
                image_path=result.image_path,
                error=result.error,
                model_used=result.model_used,
                generation_time=result.generation_time,
            )
        else:
            # Use existing pipeline - delegate to /generate endpoint behavior
            raise HTTPException(
                status_code=400,
                detail="use_extended=True or model required for extended pipeline"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Extended image generation error: {e}")
        return ImageGenerationExtendedResponse(
            success=False,
            error=str(e),
            model_used="error",
            generation_time=time.time() - start_time,
        )


# =============================================================================
# VIDEO GENERATION
# =============================================================================

@router.post("", response_model=VideoGenerationResponse)
async def generate_video_endpoint(
    background_tasks: BackgroundTasks,
    image: Optional[UploadFile] = File(None, description="Input image (optional)"),
    prompt: str = Form(..., description="Text prompt for video generation"),
    negative_prompt: str = Form(default="", description="Negative prompt"),
    width: int = Form(default=576, ge=256, le=2048, description="Frame width"),
    height: int = Form(default=576, ge=256, le=2048, description="Frame height"),
    duration: float = Form(default=2.0, ge=1.0, le=60.0, description="Video duration"),
    frames: int = Form(default=24, ge=1, le=240, description="Number of frames"),
    fps: int = Form(default=12, ge=1, le=60, description="Frames per second"),
    steps: int = Form(default=25, ge=1, le=100, description="Inference steps"),
    cfg: float = Form(default=7.0, ge=1.0, le=20.0, description="Guidance scale"),
    seed: int = Form(default=-1, description="Random seed"),
    motion_model: str = Form(default="auto", description="Motion model (auto, svd, lynx_one, polaris)"),
    sync: bool = Form(default=False, description="Run synchronously"),
) -> VideoGenerationResponse:
    """
    Generate video from text prompt.
    
    NEW endpoint - does NOT modify existing /generate endpoint.
    
    Pipeline:
    1. Generate base image or use uploaded image
    2. Apply motion using available model (SVD, Lynx, Polaris)
    3. Generate frame sequence with interpolation
    4. Encode video with ffmpeg
    """
    start_time = time.time()
    
    try:
        # Save uploaded image if present
        image_path = None
        if image:
            from extensions.video_ext import _save_upload_file
            image_path = await _save_upload_file(image, str(uuid.uuid4())[:8])
        
        # Import video generation
        from extensions.video_ext import generate_video
        
        # Generate video
        result = generate_video(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            duration=duration,
            frames=frames,
            fps=fps,
            steps=steps,
            cfg=cfg,
            seed=seed,
            motion_model=motion_model,
        )
        
        return VideoGenerationResponse(
            success=result.success,
            video_path=result.video_path,
            error=result.error,
            model_used=result.model_used,
            generation_time=result.generation_time,
        )
    
    except Exception as e:
        logger.error(f"Video generation error: {e}")
        return VideoGenerationResponse(
            success=False,
            error=str(e),
            model_used="error",
            generation_time=time.time() - start_time,
        )


# =============================================================================
# HEALTH CHECK
# =============================================================================

@router.get("/health")
async def health_check():
    """Health check for video extension."""
    try:
        # Check available models
        from extensions.models import SafeModelLoader
        loader = SafeModelLoader()
        models = loader.get_available_models()
        
        # Check VRAM
        vram = loader.get_vram_info()
        
        return {
            "status": "healthy",
            "models_available": len(models),
            "cuda_available": vram.get("cuda_available", False),
            "device": vram.get("device_used", "cpu"),
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e),
        }


# =============================================================================
# STANDALONE APP
# =============================================================================

app = FastAPI(
    title="Picture-Aliver Video Extension API",
    description="Video generation endpoint (standalone)",
    version="1.0.0",
)

app.include_router(router)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logger.info("Starting Video Extension API on port 8001")
    
    uvicorn.run(
        "extensions.api_integration:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
    )