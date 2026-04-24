"""
Picture-Aliver FastAPI Backend Server

Provides REST API for the Picture-Aliver image-to-video pipeline.
Run with: uvicorn src.picture_aliver.api:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import sys
import time
import uuid
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("picture_aliver_api")


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class GenerationRequest(BaseModel):
    """Request model for generation task."""
    prompt: str
    duration: float = 3.0
    fps: int = 8
    width: int = 512
    height: int = 512
    guidance_scale: float = 7.5
    motion_strength: float = 0.8
    motion_mode: str = "auto"
    enable_quality_check: bool = True
    enable_debug: bool = False


class GenerationResponse(BaseModel):
    """Response model for generation task."""
    task_id: str
    status: str
    message: str
    video_url: Optional[str] = None


class TaskStatus(BaseModel):
    """Status model for task tracking."""
    task_id: str
    status: str
    progress: float
    message: str
    video_path: Optional[str] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None


# =============================================================================
# TASK MANAGEMENT
# =============================================================================

class TaskManager:
    """
    Manages generation tasks and their status.
    
    Thread-safe task storage with status tracking for
    background processing.
    """
    
    def __init__(self):
        self.tasks: dict[str, dict] = {}
        self._lock = None  # Will use threading.Lock if needed
    
    def create_task(self, task_id: str) -> dict:
        """Create a new task entry."""
        self.tasks[task_id] = {
            "status": "pending",
            "progress": 0.0,
            "message": "Task created",
            "video_path": None,
            "error": None,
            "created_at": datetime.now(),
            "completed_at": None,
            "processing_time": None
        }
        return self.tasks[task_id]
    
    def update_task(
        self,
        task_id: str,
        status: str,
        progress: float,
        message: str,
        video_path: Optional[str] = None,
        error: Optional[str] = None
    ) -> None:
        """Update task status."""
        if task_id in self.tasks:
            self.tasks[task_id]["status"] = status
            self.tasks[task_id]["progress"] = progress
            self.tasks[task_id]["message"] = message
            if video_path:
                self.tasks[task_id]["video_path"] = video_path
            if error:
                self.tasks[task_id]["error"] = error
            if status == "completed":
                self.tasks[task_id]["completed_at"] = datetime.now()
    
    def get_task(self, task_id: str) -> Optional[dict]:
        """Get task by ID."""
        return self.tasks.get(task_id)
    
    def get_all_tasks(self) -> list[dict]:
        """Get all tasks."""
        return list(self.tasks.values())


# Global task manager
task_manager = TaskManager()


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="Picture-Aliver API",
    description="AI Image-to-Video Generation Pipeline API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_upload_dir() -> Path:
    """Get upload directory for input images."""
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    return upload_dir


def get_output_dir() -> Path:
    """Get output directory for generated videos."""
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    return output_dir


async def save_upload_file(upload_file: UploadFile, task_id: str) -> Path:
    """
    Save uploaded file to disk.
    
    Args:
        upload_file: FastAPI UploadFile object
        task_id: Unique task identifier
        
    Returns:
        Path to saved file
    """
    upload_dir = get_upload_dir()
    
    # Get file extension
    filename = upload_file.filename or "image"
    ext = Path(filename).suffix or ".png"
    
    # Create unique filename
    safe_filename = f"{task_id}{ext}"
    file_path = upload_dir / safe_filename
    
    # Write file content
    content = await upload_file.read()
    with open(file_path, "wb") as f:
        f.write(content)
    
    logger.info(f"Saved upload: {file_path}")
    return file_path


def run_pipeline_generation(
    task_id: str,
    image_path: Path,
    prompt: str,
    duration: float,
    fps: int,
    width: int,
    height: int,
    guidance_scale: float,
    motion_strength: float,
    motion_mode: str,
    enable_quality_check: bool,
    enable_debug: bool
) -> dict:
    """
    Run the pipeline generation synchronously.
    
    This function is called by the background task handler.
    Updates task status throughout the process.
    
    Args:
        task_id: Unique task identifier
        image_path: Path to input image
        prompt: Text prompt for generation
        duration: Video duration in seconds
        fps: Frames per second
        width: Frame width
        height: Frame height
        guidance_scale: Guidance scale for diffusion
        motion_strength: Motion strength (0.0-1.0)
        motion_mode: Motion mode (auto, cinematic, zoom, etc.)
        enable_quality_check: Enable auto-correction
        enable_debug: Enable debug output
        
    Returns:
        Dictionary with result information
    """
    start_time = time.time()
    result_info = {"success": False, "video_path": None, "error": None}
    
    try:
        logger.info(f"[{task_id}] Starting pipeline generation")
        task_manager.update_task(task_id, "processing", 0.1, "Loading pipeline...")
        
        # Import pipeline components (lazy import to avoid startup issues)
        from picture_aliver.main import (
            Pipeline, PipelineConfig, DebugConfig, run_pipeline
        )
        
        # Configure debug if enabled
        debug_config = None
        if enable_debug:
            debug_config = DebugConfig(
                enabled=True,
                directory=f"./debug/{task_id}",
                save_depth_maps=True,
                save_segmentation_masks=True,
                save_raw_frames=True,
                save_stabilized_frames=True,
                save_motion_fields=True
            )
        
        # Create pipeline configuration
        config = PipelineConfig(
            duration_seconds=duration,
            fps=fps,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            motion_strength=motion_strength,
            motion_mode=motion_mode,
            enable_quality_check=enable_quality_check,
            debug=debug_config or DebugConfig()
        )
        
        task_manager.update_task(task_id, "processing", 0.2, "Initializing models...")
        
        # Create and initialize pipeline
        pipeline = Pipeline(config)
        pipeline.initialize()
        
        # Create output path
        output_dir = get_output_dir()
        output_filename = f"{task_id}_video.mp4"
        output_path = output_dir / output_filename
        
        task_manager.update_task(task_id, "processing", 0.3, "Running image-to-video generation...")
        
        # Run the pipeline
        logger.info(f"[{task_id}] Running pipeline with image: {image_path}")
        result = pipeline.run_pipeline(
            image_path=image_path,
            prompt=prompt,
            config=config,
            output_path=output_path
        )
        
        task_manager.update_task(task_id, "processing", 0.9, "Finalizing output...")
        
        if result.success:
            processing_time = time.time() - start_time
            logger.info(f"[{task_id}] Generation complete in {processing_time:.2f}s")
            
            task_manager.update_task(
                task_id,
                "completed",
                1.0,
                "Generation successful",
                video_path=str(output_path)
            )
            
            result_info = {
                "success": True,
                "video_path": str(output_path),
                "processing_time": processing_time
            }
        else:
            error_msg = "; ".join(result.errors) if result.errors else "Unknown error"
            logger.error(f"[{task_id}] Generation failed: {error_msg}")
            
            task_manager.update_task(
                task_id,
                "failed",
                1.0,
                "Generation failed",
                error=error_msg
            )
            
            result_info = {"success": False, "error": error_msg}
        
    except Exception as e:
        error_msg = str(e)
        logger.exception(f"[{task_id}] Pipeline error: {error_msg}")
        
        task_manager.update_task(
            task_id,
            "failed",
            1.0,
            "Pipeline error",
            error=error_msg
        )
        
        result_info = {"success": False, "error": error_msg}
    
    return result_info


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint - API health check."""
    return {
        "name": "Picture-Aliver API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    gpu_info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    if torch.cuda.is_available():
        gpu_info["device_name"] = torch.cuda.get_device_name(0)
        gpu_info["total_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_info["memory_allocated_gb"] = torch.cuda.memory_allocated(0) / (1024**3)
    
    return {
        "status": "healthy",
        "gpu": gpu_info,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get status of a specific task."""
    task = task_manager.get_task(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    return TaskStatus(
        task_id=task_id,
        status=task["status"],
        progress=task["progress"],
        message=task["message"],
        video_path=task.get("video_path"),
        error=task.get("error"),
        processing_time=task.get("processing_time")
    )


@app.get("/tasks")
async def list_tasks():
    """List all tasks."""
    tasks = task_manager.get_all_tasks()
    return {
        "count": len(tasks),
        "tasks": [
            {
                "task_id": tid,
                "status": t["status"],
                "progress": t["progress"],
                "created_at": t.get("created_at").isoformat() if t.get("created_at") else None
            }
            for tid, t in task_manager.tasks.items()
        ]
    }


@app.post("/generate", response_model=GenerationResponse)
async def generate_video(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(..., description="Input image file"),
    prompt: str = Form(..., description="Text prompt for video generation"),
    duration: float = Form(default=3.0, ge=1.0, le=60.0, description="Video duration in seconds"),
    fps: int = Form(default=8, ge=1, le=60, description="Frames per second"),
    width: int = Form(default=512, ge=256, le=2048, description="Frame width"),
    height: int = Form(default=512, ge=256, le=2048, description="Frame height"),
    guidance_scale: float = Form(default=7.5, ge=1.0, le=20.0, description="Guidance scale"),
    motion_strength: float = Form(default=0.8, ge=0.0, le=1.0, description="Motion strength"),
    motion_mode: str = Form(default="auto", description="Motion mode"),
    enable_quality_check: bool = Form(default=True, description="Enable auto-correction"),
    sync: bool = Form(default=False, description="Run synchronously (blocks response)")
):
    """
    Generate a video from an image.
    
    This endpoint:
    1. Saves the uploaded image
    2. Creates a background task
    3. Returns immediately with a task_id (async mode)
    4. Or waits for completion and returns video URL (sync mode)
    
    For async mode, poll /tasks/{task_id} for status.
    
    Args:
        image: Input image file (PNG, JPG, etc.)
        prompt: Text prompt describing desired motion/animation
        duration: Video duration in seconds (1-60)
        fps: Frames per second (1-60)
        width: Output frame width (256-2048)
        height: Output frame height (256-2048)
        guidance_scale: How closely to follow the prompt (1-20)
        motion_strength: Motion intensity (0-1)
        motion_mode: auto, cinematic, zoom, pan, subtle, furry
        enable_quality_check: Enable automatic failure detection/correction
        sync: If true, wait for completion before returning
    
    Returns:
        GenerationResponse with task_id and status
    """
    # Generate unique task ID
    task_id = str(uuid.uuid4())[:8]
    
    logger.info(f"[{task_id}] New generation request: prompt='{prompt}', duration={duration}s")
    
    try:
        # Validate file type
        allowed_types = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
        if image.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {image.content_type}. Allowed: {allowed_types}"
            )
        
        # Save uploaded file
        image_path = await save_upload_file(image, task_id)
        
        # Create task entry
        task_manager.create_task(task_id)
        
        if sync:
            # Synchronous mode - wait for completion
            logger.info(f"[{task_id}] Running in synchronous mode")
            task_manager.update_task(task_id, "processing", 0.1, "Starting generation...")
            
            result = run_pipeline_generation(
                task_id=task_id,
                image_path=image_path,
                prompt=prompt,
                duration=duration,
                fps=fps,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                motion_strength=motion_strength,
                motion_mode=motion_mode,
                enable_quality_check=enable_quality_check,
                enable_debug=False
            )
            
            if result["success"]:
                video_url = f"/download/{task_id}"
                return GenerationResponse(
                    task_id=task_id,
                    status="completed",
                    message="Video generated successfully",
                    video_url=video_url
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Generation failed: {result['error']}"
                )
        else:
            # Async mode - add background task
            logger.info(f"[{task_id}] Running in asynchronous mode")
            
            background_tasks.add_task(
                run_pipeline_generation,
                task_id=task_id,
                image_path=image_path,
                prompt=prompt,
                duration=duration,
                fps=fps,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                motion_strength=motion_strength,
                motion_mode=motion_mode,
                enable_quality_check=enable_quality_check,
                enable_debug=False
            )
            
            return GenerationResponse(
                task_id=task_id,
                status="pending",
                message=f"Generation started. Poll /tasks/{task_id} for status."
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[{task_id}] Error starting generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download/{task_id}")
async def download_video(task_id: str):
    """
    Download the generated video for a task.
    
    Args:
        task_id: Task identifier
        
    Returns:
        Video file or 404 if not ready/found
    """
    task = task_manager.get_task(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    if task["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Task not completed. Status: {task['status']}"
        )
    
    video_path = task.get("video_path")
    if not video_path or not Path(video_path).exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    
    return FileResponse(
        path=video_path,
        media_type="video/mp4",
        filename=f"{task_id}_video.mp4"
    )


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def start_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """
    Start the FastAPI server.
    
    Args:
        host: Server host address
        port: Server port
        reload: Enable auto-reload for development
    """
    logger.info(f"Starting Picture-Aliver API server on {host}:{port}")
    logger.info(f"API docs available at http://{host}:{port}/docs")
    
    uvicorn.run(
        "picture_aliver.api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Picture-Aliver API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    start_server(host=args.host, port=args.port, reload=args.reload)