"""
Video Extension Pipeline

NEW endpoint ONLY: POST /generate-video

Do NOT modify existing endpoints.

Pipeline:
1. Generate base image (reuse existing pipeline if possible)
2. Apply motion using FREE methods:
   - AnimateDiff (preferred if available)
   - OR simple latent noise progression fallback
3. Generate frame sequence
4. Run interpolation (basic implementation acceptable)
5. Encode video via ffmpeg
"""

from __future__ import annotations

import os
import sys
import time
import logging
import subprocess
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from PIL import Image
import numpy as np

logger = logging.getLogger("picture_aliver.extensions.video")


# =============================================================================
# Result Types
# =============================================================================

@dataclass
class VideoResult:
    """Result of video generation."""
    success: bool
    video_path: Optional[str] = None
    frames: Optional[List[Image.Image]] = None
    error: Optional[str] = None
    model_used: str = "default"
    generation_time: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# =============================================================================
# Utility Functions
# =============================================================================

async def _save_upload_file(upload_file: UploadFile, task_id: str) -> str:
    """Save uploaded file to temp directory."""
    import tempfile
    import aiofiles
    
    temp_dir = Path(tempfile.gettempdir()) / "picture_aliver"
    temp_dir.mkdir(exist_ok=True)
    
    file_path = temp_dir / f"{task_id}_{upload_file.filename}"
    
    async with aiofiles.open(file_path, "wb") as f:
        content = await upload_file.read()
        await f.write(content)
    
    return str(file_path)
    """Result of video generation."""
    success: bool
    video_path: Optional[str] = None
    frames: Optional[List[Image.Image]] = None
    error: Optional[str] = None
    model_used: str = "default"
    generation_time: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# =============================================================================
# Video Generation Function
# =============================================================================

async def generate_video(
    prompt: str,
    # Generation params
    negative_prompt: str = "",
    width: int = 576,
    height: int = 576,
    duration: float = 2.0,
    frames: int = 24,
    fps: int = 12,
    steps: int = 25,
    cfg: float = 7.0,
    seed: int = -1,
    # Model selection
    motion_model: str = "auto",  # svd, lynx_one, polaris, auto
    # Output
    output_path: Optional[str] = None,
    # Existing pipeline fallback
    existing_generator=None,
) -> VideoResult:
    """
    Generate video from prompt.
    
    Args:
        prompt: Text prompt
        negative_prompt: Negative prompt
        width: Frame width
        height: Frame height
        duration: Video duration in seconds
        frames: Number of frames to generate
        fps: Frames per second
        steps: Number of inference steps
        cfg: Guidance scale
        seed: Random seed (-1 for random)
        motion_model: Motion model to use (auto, svd, lynx_one, polaris)
        output_path: Path to save video
        existing_generator: Fallback generator
        
    Returns:
        VideoResult with video path and frames
    """
    import asyncio
    return asyncio.run(_generate_video_async(
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
        output_path=output_path,
        existing_generator=existing_generator,
    ))


async def _generate_video_async(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    duration: float,
    frames: int,
    fps: int,
    steps: int,
    cfg: float,
    seed: int,
    motion_model: str,
    output_path: Optional[str],
    existing_generator,
) -> VideoResult:
    """Async video generation."""
    start_time = time.time()
    
    # Determine motion model
    selected_model = _select_motion_model(motion_model)
    
    logger.info(f"Using motion model: {selected_model}")
    
    # Try generating with motion model
    if selected_model:
        try:
            result = await _generate_with_motion_model(
                model_id=selected_model,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                frames=frames,
                steps=steps,
                cfg=cfg,
                seed=seed,
            )
            
            if result.success:
                # Encode to video
                if result.frames:
                    video_path = await _encode_video(
                        frames=result.frames,
                        fps=fps,
                        output_path=output_path,
                    )
                    result.video_path = video_path
                
                result.generation_time = time.time() - start_time
                return result
            
            logger.warning(f"Motion model {selected_model} failed: {result.error}")
            
        except Exception as e:
            logger.warning(f"Motion generation error: {e}")
    
    # Fallback: Generate base image + latent interpolation
    logger.info("Using latent interpolation fallback")
    
    return await _generate_latent_fallback(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        frames=frames,
        fps=fps,
        output_path=output_path,
        existing_generator=existing_generator,
    )


def _select_motion_model(model: str) -> Optional[str]:
    """Select best available motion model."""
    if model == "auto":
        # Check what's available
        try:
            from .models import SafeModelLoader
            loader = SafeModelLoader()
            available = loader.get_available_models()
            
            # Prefer these in order
            for pref in ["svd", "lynx_one", "polaris"]:
                if any(pref in m.id.lower() for m in available):
                    return pref
            
            # Any motion model
            for m in available:
                if m.category.value == "motion":
                    return m.id
            
        except Exception:
            pass
        
        return None
    
    return model if model != "auto" else None


async def _generate_with_motion_model(
    model_id: str,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    frames: int,
    steps: int,
    cfg: float,
    seed: int,
) -> VideoResult:
    """Generate using motion model with performance optimizations."""
    try:
        from .models import SafeModelLoader
        
        loader = SafeModelLoader()
        
        # Performance: Use autocast if GPU available
        vram = loader.get_vram_info()
        use_fp16 = vram.get("cuda_available", False)
        
        # Set torch settings for performance
        if use_fp16 and torch.cuda.is_available():
            with torch.autocast("cuda", dtype=torch.float16):
                result = await loader.generate_video(
                    prompt=prompt,
                    model_id=model_id,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_frames=frames,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    seed=seed,
                )
        else:
            # CPU fallback
            result = await loader.generate_video(
                prompt=prompt,
                model_id=model_id,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_frames=frames,
                num_inference_steps=steps,
                guidance_scale=cfg,
                seed=seed,
            )
        
        return VideoResult(
            success=result.success,
            frames=result.frames if result.success else None,
            error=result.error,
            model_used=model_id,
        )
        
    except Exception as e:
        return VideoResult(
            success=False,
            error=str(e),
            model_used=model_id,
        )


async def _generate_latent_fallback(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    frames: int,
    fps: int,
    output_path: Optional[str],
    existing_generator,
) -> VideoResult:
    """Generate base image + latent interpolation fallback."""
    try:
        # Generate base image using existing pipeline
        from .image_ext import generate_image_extended
        
        base_result = generate_image_extended(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=20,
            save_path=None,  # Don't save yet
        )
        
        if not base_result.success or not base_result.image:
            return VideoResult(
                success=False,
                error=f"Base image generation failed: {base_result.error}",
            )
        
        # Create interpolated frames
        frames_list = _create_latent_frames(
            base_image=base_result.image,
            num_frames=frames,
        )
        
        # Encode to video
        video_path = await _encode_video(
            frames=frames_list,
            fps=fps,
            output_path=output_path,
        )
        
        return VideoResult(
            success=True,
            video_path=video_path,
            frames=frames_list,
            model_used="latent_interpolation",
        )
        
    except Exception as e:
        return VideoResult(
            success=False,
            error=f"Latent fallback error: {e}",
        )


def _create_latent_frames(
    base_image: Image.Image,
    num_frames: int,
) -> List[Image.Image]:
    """Create frames with subtle motion via latent interpolation."""
    frames = []
    
    # Resize for generation
    base = base_image.resize((512, 512), Image.LANCZOS)
    arr = np.array(base, dtype=np.float32) / 255.0
    
    for i in range(num_frames):
        # Subtle noise-based motion
        t = i / max(num_frames - 1, 1)
        
        # Progressive noise pattern
        noise = np.random.randn(*arr.shape) * 0.03 * (1 - t)
        
        # Blend with shifted pixels (subtle motion effect)
        shift = int(3 * (0.5 - abs(t - 0.5)))  # Move toward center
        
        if shift > 0:
            shifted = np.roll(arr, shift, axis=(0 if i % 2 else 1))
        else:
            shifted = arr
        
        # Apply noise
        frame_arr = shifted + noise
        frame_arr = np.clip(frame_arr, 0, 1)
        
        frame = Image.fromarray((frame_arr * 255).astype(np.uint8))
        frames.append(frame)
    
    return frames


async def _encode_video(
    frames: List[Image.Image],
    fps: int,
    output_path: Optional[str],
) -> Optional[str]:
    """Encode frames to video using ffmpeg."""
    # Create temp directory
    import tempfile
    import shutil
    import asyncio
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Save frames
        frame_dir = Path(temp_dir) / "frames"
        frame_dir.mkdir()
        
        for i, frame in enumerate(frames):
            frame.save(frame_dir / f"frame_{i:04d}.png")
        
        # Default output path
        if not output_path:
            output_dir = Path("outputs")
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"video_{int(time.time())}.mp4"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use ffmpeg
        input_pattern = str(frame_dir / "frame_%04d.png")
        
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", input_pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "23",
            str(output_path)
        ]
        
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        
        stdout, stderr = await proc.communicate()
        
        if proc.returncode != 0:
            logger.warning(f"ffmpeg error: {stderr.decode()}")
            # Return frames as GIF fallback
            output_path = output_path.with_suffix(".gif")
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=int(1000/fps),
                loop=0,
            )
        
        return str(output_path)
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _create_interpolated_frames(
    frames: List[Image.Image],
    target_fps: int,
) -> List[Image.Image]:
    """Create smooth interpolated frames using simple blending."""
    if len(frames) < 2:
        return frames
    
    interpolated = []
    
    for i in range(len(frames) - 1):
        frame1 = frames[i]
        frame2 = frames[i + 1]
        
        # Add current frame
        interpolated.append(frame1)
        
        # Calculate blend count based on desired FPS
        blend_count = max(1, target_fps // 2)
        
        for j in range(1, blend_count + 1):
            alpha = j / (blend_count + 1)
            
            # Blend frames
            blended = Image.blend(frame1, frame2, alpha)
            interpolated.append(blended)
    
    # Add final frame
    interpolated.append(frames[-1])
    
    return interpolated


def _smooth_frames_cv(
    frames: List[Image.Image],
    kernel_size: int = 3,
) -> List[Image.Image]:
    """Smooth frames using OpenCV (optional, graceful fallback if unavailable)."""
    try:
        import cv2
        import numpy as np
        
        smoothed = []
        
        for frame in frames:
            arr = np.array(frame)
            
            # Apply Gaussian blur for smoothing
            blurred = cv2.GaussianBlur(arr, (kernel_size, kernel_size), 0)
            
            smoothed.append(Image.fromarray(blurred))
        
        return smoothed
        
    except ImportError:
        logger.warning("OpenCV not available, using PIL-based smoothing")
        return frames


def _generate_video_sync(
    prompt: str,
    **kwargs
) -> VideoResult:
    """Synchronous wrapper for generate_video."""
    return generate_video(prompt, **kwargs)


# =============================================================================
# FastAPI Endpoint
# =============================================================================

def create_video_endpoint():
    """Create FastAPI endpoint for video generation."""
    from fastapi import APIRouter, UploadOption
    from pydantic import BaseModel
    
    router = APIRouter(prefix="/generate-video", tags=["video"])
    
    class VideoRequest(BaseModel):
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
        output_path: Optional[str] = None
    
    class VideoResponse(BaseModel):
        success: bool
        video_path: Optional[str] = None
        error: Optional[str] = None
        model_used: str
        generation_time: float
    
    @router.post("", response_model=VideoResponse)
    async def generate_video_endpoint(request: VideoRequest):
        result = await _generate_video_async(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            duration=request.duration,
            frames=request.frames,
            fps=request.fps,
            steps=request.steps,
            cfg=request.cfg,
            seed=request.seed,
            motion_model=request.motion_model,
            output_path=request.output_path,
            existing_generator=None,
        )
        
        return VideoResponse(
            success=result.success,
            video_path=result.video_path,
            error=result.error,
            model_used=result.model_used,
            generation_time=result.generation_time,
        )
    
    return router


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    print("=== Video Extension Test ===\n")
    
    # Test video generation
    result = generate_video(
        "A flowing river with mountains in background",
        width=512,
        height=512,
        frames=12,
        fps=12,
    )
    
    print(f"Success: {result.success}")
    print(f"Video: {result.video_path}")
    print(f"Model: {result.model_used}")
    print(f"Error: {result.error}")
    print(f"Time: {result.generation_time:.2f}s")