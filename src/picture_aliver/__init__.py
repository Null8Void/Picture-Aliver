"""
Picture-Aliver: AI-Powered Image to Video Animation System

Complete pipeline for converting images to animated videos.
Supports all content types including human, furry, landscape, and objects.
"""

from __future__ import annotations

from .image_loader import ImageLoader
from .depth_estimator import DepthEstimator, DepthResult
from .segmentation import SegmentationModule, SegmentationResult, ContentType
from .motion_generator import (
    MotionGenerator, MotionField, MotionMode, CameraTrajectory, FurryMotionGenerator
)
from .motion_prompt import (
    MotionPromptParser, MotionPromptMapper, MotionParameters, 
    MotionCategory, MotionIntensity
)
from .video_generator import VideoGenerator, VideoFrames, GenerationConfig
from .text_to_image import TextToImageGenerator, TextToVideoGenerator, T2IConfig
from .stabilizer import VideoStabilizer, StabilizationConfig
from .exporter import (
    VideoExporter, ExportConfig, VideoFormat, VideoSpec, ExportOptions, 
    QualityPreset, Codec, export_video
)
from .quality_control import (
    QualityController, QualityReport, QualityDetector, QualityIssue,
    assess_video_quality
)
from .gpu_optimization import (
    GPUOptimizer, VRAMTier, GPUConfig, ModelOffloader,
    optimize_model_for_device, print_benchmark_table
)
from .main import Pipeline, PipelineConfig, PipelineResult, run_pipeline

__version__ = "1.0.0"

__all__ = [
    "ImageLoader",
    "DepthEstimator",
    "DepthResult",
    "SegmentationModule",
    "SegmentationResult",
    "ContentType",
    "MotionGenerator",
    "MotionField",
    "MotionMode",
    "CameraTrajectory",
    "FurryMotionGenerator",
    "MotionPromptParser",
    "MotionPromptMapper",
    "MotionParameters",
    "MotionCategory",
    "MotionIntensity",
    "VideoGenerator",
    "VideoFrames",
    "GenerationConfig",
    "TextToImageGenerator",
    "TextToVideoGenerator",
    "T2IConfig",
    "VideoStabilizer",
    "StabilizationConfig",
    "VideoExporter",
    "ExportConfig",
    "VideoFormat",
    "VideoSpec",
    "ExportOptions",
    "QualityPreset",
    "Codec",
    "export_video",
    "QualityController",
    "QualityReport",
    "QualityDetector",
    "QualityIssue",
    "assess_video_quality",
    "GPUOptimizer",
    "VRAMTier",
    "GPUConfig",
    "ModelOffloader",
    "optimize_model_for_device",
    "print_benchmark_table",
    "Pipeline",
    "PipelineConfig",
    "PipelineResult",
    "run_pipeline",
]

__doc__ = """
Picture-Aliver: Complete AI Image-to-Video Pipeline
==================================================

Quick Start:
    from picture_aliver import run_pipeline, PipelineConfig
    
    result = run_pipeline(
        image_path="input.jpg",
        output_path="output.mp4",
        config=PipelineConfig(duration_seconds=10, fps=24)
    )

Pipeline Steps:
    1. Image Loader
    2. Depth Estimation (MiDaS)
    3. Segmentation (SAM)
    4. Motion Generator
    5. Video Diffusion
    6. Stabilization
    7. Frame Interpolation
    8. Export (FFmpeg)

GPU Requirements:
    - Minimum: 2GB VRAM (low resolution, short clips)
    - Recommended: 8GB VRAM (720p, 30s clips)
    - Optimal: 12GB+ VRAM (1080p, 60s+ clips)
"""