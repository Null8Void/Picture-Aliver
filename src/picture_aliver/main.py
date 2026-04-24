"""
Picture-Aliver: Complete AI Image-to-Video Pipeline

Main entry point with run_pipeline() orchestrating all modules.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Optional, Union, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

import torch
import numpy as np
from PIL import Image

from image_loader import ImageLoader
from depth_estimator import DepthEstimator, DepthResult
from segmentation import SegmentationModule, SegmentationResult, ContentType
from motion_generator import MotionGenerator, MotionField
from video_generator import VideoGenerator, VideoFrames
from stabilizer import VideoStabilizer
from text_to_image import TextToImageGenerator, TextToVideoGenerator
from quality_control import QualityController, QualityReport
from gpu_optimization import GPUOptimizer
from exporter import VideoExporter, ExportOptions, VideoSpec, QualityPreset, VideoFormat


@dataclass
class DebugConfig:
    """Debug system configuration."""
    enabled: bool = False
    directory: str = "./debug"
    save_depth_maps: bool = True
    save_segmentation_masks: bool = True
    save_raw_frames: bool = True
    save_stabilized_frames: bool = True
    save_motion_fields: bool = True
    format: str = "png"
    frame_interval: int = 1


class DebugSaver:
    """Save intermediate pipeline outputs for debugging."""
    
    def __init__(self, config: DebugConfig, run_id: Optional[str] = None):
        self.config = config
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(config.directory) / run_id
        self._created = False
    
    def _ensure_dir(self) -> None:
        if not self._created:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            (self.run_dir / "depth").mkdir(exist_ok=True)
            (self.run_dir / "segmentation").mkdir(exist_ok=True)
            (self.run_dir / "frames_raw").mkdir(exist_ok=True)
            (self.run_dir / "frames_stabilized").mkdir(exist_ok=True)
            (self.run_dir / "motion").mkdir(exist_ok=True)
            self._created = True
    
    def save_depth_map(self, depth: torch.Tensor, step: int = 1) -> None:
        if not self.config.enabled or not self.config.save_depth_maps:
            return
        self._ensure_dir()
        try:
            depth_np = depth.detach().cpu().numpy()
            if depth_np.ndim == 3:
                depth_np = depth_np[0]
            depth_normalized = ((depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8) * 255).astype(np.uint8)
            Image.fromarray(depth_normalized).save(self.run_dir / "depth" / f"depth_step{step:02d}.{self.config.format}")
            print(f"  [Debug] Saved depth map: depth/step{step:02d}.{self.config.format}")
        except Exception as e:
            print(f"  [Debug] Failed to save depth: {e}")
    
    def save_segmentation(self, segmentation: SegmentationResult, step: int = 1) -> None:
        if not self.config.enabled or not self.config.save_segmentation_masks:
            return
        self._ensure_dir()
        try:
            if hasattr(segmentation, 'mask') and segmentation.mask is not None:
                mask = segmentation.mask.detach().cpu().numpy()
                if mask.ndim == 3:
                    mask = mask[0]
                mask_normalized = (mask * 255).astype(np.uint8)
                Image.fromarray(mask_normalized).save(self.run_dir / "segmentation" / f"mask_step{step:02d}.{self.config.format}")
                print(f"  [Debug] Saved segmentation: segmentation/mask_step{step:02d}.{self.config.format}")
        except Exception as e:
            print(f"  [Debug] Failed to save segmentation: {e}")
    
    def save_frames(self, frames: VideoFrames, prefix: str, step: int = 1) -> None:
        if not self.config.enabled:
            return
        is_stabilized = "stabilized" in prefix
        if is_stabilized and not self.config.save_stabilized_frames:
            return
        if not is_stabilized and not self.config.save_raw_frames:
            return
        self._ensure_dir()
        try:
            frame_list = frames.to_list() if hasattr(frames, 'to_list') else []
            if not frame_list and hasattr(frames, 'frames'):
                frame_list = [f.detach().cpu().numpy().transpose(1, 2, 0) if f.ndim == 3 else f.detach().cpu().numpy() for f in frames.frames]
            for i, frame in enumerate(frame_list):
                if i % self.config.frame_interval != 0:
                    continue
                if frame.ndim == 3 and frame.shape[0] == 3:
                    frame = frame.transpose(1, 2, 0)
                frame_uint8 = (np.clip(frame, 0, 1) * 255).astype(np.uint8) if frame.max() <= 1 else frame.astype(np.uint8)
                Image.fromarray(frame_uint8).save(self.run_dir / f"frames_{prefix}" / f"frame_{i:04d}.{self.config.format}")
            print(f"  [Debug] Saved {len(frame_list)} frames: frames_{prefix}/")
        except Exception as e:
            print(f"  [Debug] Failed to save frames: {e}")
    
    def save_motion_field(self, motion: MotionField, step: int = 1) -> None:
        if not self.config.enabled or not self.config.save_motion_fields:
            return
        self._ensure_dir()
        try:
            if hasattr(motion, 'flows') and motion.flows:
                flow = motion.flows[-1]
                if isinstance(flow, torch.Tensor):
                    flow = flow.cpu().numpy()
                flow_vis = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
                magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                angle = np.arctan2(flow[..., 1], flow[..., 0])
                flow_vis[..., 0] = ((angle + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
                flow_vis[..., 1] = (np.clip(magnitude / (magnitude.max() + 1e-8) * 255, 0, 255)).astype(np.uint8)
                flow_vis[..., 2] = 255
                Image.fromarray(flow_vis).save(self.run_dir / "motion" / f"flow_step{step:02d}.{self.config.format}")
                print(f"  [Debug] Saved motion field: motion/flow_step{step:02d}.{self.config.format}")
        except Exception as e:
            print(f"  [Debug] Failed to save motion: {e}")


@dataclass
class PipelineConfig:
    """Configuration for the complete pipeline."""
    duration_seconds: float = 3.0
    fps: int = 8
    width: int = 512
    height: int = 512
    guidance_scale: float = 7.5
    num_inference_steps: int = 25
    motion_strength: float = 0.8
    motion_mode: str = "auto"
    motion_prompt: Optional[str] = None
    quality: str = "medium"
    output_format: str = "mp4"
    enable_stabilization: bool = True
    enable_interpolation: bool = False
    enable_quality_check: bool = True
    quality_max_retries: int = 2
    device: Optional[str] = None
    model_dir: Optional[Path] = None
    debug: DebugConfig = field(default_factory=DebugConfig)


@dataclass
class PipelineResult:
    """Result from pipeline execution."""
    success: bool = False
    output_path: Optional[Path] = None
    duration_seconds: float = 0.0
    num_frames: int = 0
    processing_time: float = 0.0
    quality_score: Optional[float] = None
    detected_content_type: Optional[str] = None
    errors: list = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class Pipeline:
    """
    Complete AI Image-to-Video pipeline.
    
    Orchestrates all modules in exact order:
    1. Image Loader -> 2. Depth Estimation -> 3. Segmentation -> 4. Motion
    -> 5. Video Diffusion -> 6. Stabilization -> 7. Interpolation -> 8. Export
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        
        self.device = self._setup_device()
        
        self.image_loader: Optional[ImageLoader] = None
        self.depth_estimator: Optional[DepthEstimator] = None
        self.segmentation: Optional[SegmentationModule] = None
        self.motion_generator: Optional[MotionGenerator] = None
        self.video_generator: Optional[VideoGenerator] = None
        self.stabilizer: Optional[VideoStabilizer] = None
        self.text_to_image: Optional[TextToImageGenerator] = None
        self.quality_controller: Optional[QualityController] = None
        self.gpu_optimizer: Optional[GPUOptimizer] = None
        self.exporter: Optional[VideoExporter] = None
        
        self.debug_saver: Optional[DebugSaver] = None
        self._initialized = False
    
    def _setup_device(self) -> torch.device:
        """Setup compute device."""
        if self.config.device:
            return torch.device(self.config.device)
        
        if torch.cuda.is_available():
            print(f"[Pipeline] Using CUDA: {torch.cuda.get_device_name(0)}")
            return torch.device("cuda")
        
        print("[Pipeline] Using CPU")
        return torch.device("cpu")
    
    def initialize(self) -> None:
        """Initialize all pipeline modules."""
        if self._initialized:
            return
        
        print("[Pipeline] Initializing modules...")
        
        self.gpu_optimizer = GPUOptimizer(device=self.device)
        self.image_loader = ImageLoader(device=self.device)
        self.depth_estimator = DepthEstimator(device=self.device, model_type="zoedepth")
        self.segmentation = SegmentationModule(device=self.device)
        self.motion_generator = MotionGenerator(device=self.device, depth_estimator=self.depth_estimator)
        self.video_generator = VideoGenerator(device=self.device, depth_estimator=self.depth_estimator)
        self.stabilizer = VideoStabilizer(device=self.device)
        self.text_to_image = TextToImageGenerator(device=self.device)
        self.quality_controller = QualityController(device=self.device, max_retries=self.config.quality_max_retries)
        self.exporter = VideoExporter(device=self.device)
        
        if self.config.debug.enabled:
            self.debug_saver = DebugSaver(self.config.debug)
            print(f"[Debug] Debug output enabled: {self.debug_saver.run_dir}")
        
        self._initialized = True
        print("[Pipeline] All modules initialized")
    
    def run_pipeline(
        self,
        image_path: Union[str, Path],
        prompt: str = "",
        config: Optional[PipelineConfig] = None,
        output_path: Optional[Union[str, Path]] = None
    ) -> PipelineResult:
        """
        Run the complete image-to-video pipeline.
        
        Args:
            image_path: Path to input image
            prompt: Text prompt for generation
            config: Pipeline configuration
            output_path: Optional output path override
            
        Returns:
            PipelineResult with output path and metadata
        """
        if config:
            self.config = config
        
        result = PipelineResult()
        start_time = time.time()
        
        try:
            if not self._initialized:
                self.initialize()
            
            image_path = Path(image_path)
            
            if output_path is None:
                output_path = Path("./output") / f"{image_path.stem}_video.{self.config.output_format}"
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            print(f"\n{'='*60}")
            print(f"PICTURE-ALIVER PIPELINE")
            print(f"{'='*60}")
            print(f"Input: {image_path}")
            print(f"Output: {output_path}")
            print(f"Duration: {self.config.duration_seconds}s @ {self.config.fps}fps")
            print(f"{'='*60}\n")
            
            step1_image = self._step1_load_image(image_path)
            print()
            
            step2_depth = self._step2_estimate_depth(step1_image)
            print()
            
            step3_segmentation = self._step3_segmentation(step1_image)
            result.detected_content_type = step3_segmentation.content_type.value
            print()
            
            step4_motion = self._step4_generate_motion(
                step1_image, step2_depth, step3_segmentation
            )
            print()
            
            step5_video = self._step5_video_diffusion(
                step1_image, step2_depth, step3_segmentation, step4_motion, prompt
            )
            print()
            
            step6_stabilized = self._step6_stabilize(step5_video, step4_motion)
            print()
            
            if self.config.enable_interpolation:
                step7_interpolated = self._step7_interpolate(step6_stabilized)
            else:
                step7_interpolated = step6_stabilized
            print()
            
            if self.config.enable_quality_check:
                step8_quality_check = self._step8_quality_check(step7_interpolated)
                if step8_quality_check:
                    result.quality_score = step8_quality_check.overall_score
                    print()
            
            self._step9_export(step7_interpolated, output_path)
            
            result.success = True
            result.output_path = output_path
            result.duration_seconds = self.config.duration_seconds
            result.num_frames = int(self.config.duration_seconds * self.config.fps)
            
            print(f"\n{'='*60}")
            print(f"PIPELINE COMPLETE")
            print(f"{'='*60}")
            print(f"Output: {output_path}")
            print(f"Duration: {result.duration_seconds}s")
            print(f"Frames: {result.num_frames}")
            if result.quality_score:
                print(f"Quality: {result.quality_score:.2f}")
            print(f"{'='*60}\n")
            
        except Exception as e:
            result.errors.append(str(e))
            print(f"\n[Pipeline] ERROR: {e}")
            traceback.print_exc()
        
        result.processing_time = time.time() - start_time
        return result
    
    def _step1_load_image(self, image_path: Path) -> torch.Tensor:
        """Step 1: Load image."""
        print(f"[Step 1/9] Image Loader")
        print(f"  Loading: {image_path}")
        image = self.image_loader.load(image_path)
        print(f"  Loaded shape: {image.shape}")
        return image
    
    def _step2_estimate_depth(self, image: torch.Tensor) -> DepthResult:
        """Step 2: Estimate depth."""
        print(f"[Step 2/9] Depth Estimation (MiDaS)")
        print(f"  Estimating depth map...")
        depth_result = self.depth_estimator.estimate(image)
        print(f"  Depth range: {depth_result.depth.min():.2f} - {depth_result.depth.max():.2f}")
        
        if self.debug_saver:
            depth_tensor = depth_result.depth if hasattr(depth_result, 'depth') else depth_result.normalized
            self.debug_saver.save_depth_map(depth_tensor, step=2)
        
        return depth_result
    
    def _step3_segmentation(self, image: torch.Tensor) -> SegmentationResult:
        """Step 3: Segmentation."""
        print(f"[Step 3/9] Segmentation (SAM)")
        print(f"  Performing semantic segmentation...")
        segmentation = self.segmentation.segment(image)
        print(f"  Detected: {segmentation.content_type.value}")
        print(f"  Categories: {segmentation.categories[:5]}...")
        
        if self.debug_saver:
            self.debug_saver.save_segmentation(segmentation, step=3)
        
        return segmentation
    
    def _step4_generate_motion(
        self,
        image: torch.Tensor,
        depth: DepthResult,
        segmentation: SegmentationResult
    ) -> MotionField:
        """Step 4: Generate motion field."""
        print(f"[Step 4/9] Motion Generator")
        print(f"  Mode: {self.config.motion_mode}")
        print(f"  Strength: {self.config.motion_strength}")
        if self.config.motion_prompt:
            print(f"  Prompt: {self.config.motion_prompt}")
        
        depth_tensor = depth.depth if hasattr(depth, 'depth') else depth.normalized
        
        motion_field = self.motion_generator.generate(
            image=image,
            depth=depth_tensor,
            segmentation=segmentation,
            mode=self.config.motion_mode,
            strength=self.config.motion_strength,
            num_frames=int(self.config.duration_seconds * self.config.fps),
            motion_prompt=self.config.motion_prompt
        )
        print(f"  Motion flows generated: {len(motion_field.flows)} frames")
        
        if self.debug_saver:
            self.debug_saver.save_motion_field(motion_field, step=4)
        
        return motion_field
    
    def _step5_video_diffusion(
        self,
        image: torch.Tensor,
        depth: DepthResult,
        segmentation: SegmentationResult,
        motion: MotionField,
        prompt: str
    ) -> VideoFrames:
        """Step 5: Video diffusion generation."""
        print(f"[Step 5/9] Video Diffusion")
        print(f"  Generating {int(self.config.duration_seconds * self.config.fps)} frames...")
        print(f"  Prompt: '{prompt or 'animated scene'}'")
        print(f"  Steps: {self.config.num_inference_steps}")
        print(f"  Guidance: {self.config.guidance_scale}")
        
        depth_tensor = depth.depth if hasattr(depth, 'depth') else depth.normalized
        
        video_frames = self.video_generator.generate(
            image_tensor=image,
            depth_map=depth_tensor,
            motion_field=motion,
            segmentation=segmentation,
            prompt=prompt or self._get_default_prompt(segmentation.content_type.value),
            num_frames=int(self.config.duration_seconds * self.config.fps),
            guidance_scale=self.config.guidance_scale,
            num_inference_steps=self.config.num_inference_steps
        )
        
        print(f"  Generated {len(video_frames)} frames")
        
        if self.debug_saver:
            self.debug_saver.save_frames(video_frames, "raw", step=5)
        
        return video_frames
    
    def _step6_stabilize(
        self,
        frames: VideoFrames,
        motion: MotionField
    ) -> VideoFrames:
        """Step 6: Stabilization."""
        print(f"[Step 6/9] Stabilization")
        
        if not self.config.enable_stabilization:
            print(f"  Skipped (disabled)")
            return frames
        
        print(f"  Applying temporal smoothing...")
        stabilized = self.stabilizer.stabilize(frames, motion_field=motion)
        print(f"  Stabilized {len(stabilized)} frames")
        
        if self.debug_saver:
            self.debug_saver.save_frames(stabilized, "stabilized", step=6)
        
        return stabilized
    
    def _step7_interpolate(self, frames: VideoFrames) -> VideoFrames:
        """Step 7: Frame interpolation."""
        print(f"[Step 7/9] Frame Interpolation (RIFE)")
        print(f"  Factor: 2x")
        
        if frames.to_tensor().dim() == 4:
            tensor = frames.to_tensor()
        else:
            tensor = frames.to_tensor()
        
        t, c, h, w = tensor.shape
        interpolated_frames = []
        
        for i in range(t - 1):
            frame1 = tensor[i]
            frame2 = tensor[i + 1]
            interpolated_frames.append(frame1)
            
            interp = (frame1 + frame2) / 2
            interpolated_frames.append(interp)
        
        interpolated_frames.append(tensor[-1])
        
        result = VideoFrames()
        result.extend(interpolated_frames)
        print(f"  Interpolated to {len(result)} frames")
        return result
    
    def _step8_quality_check(self, frames: VideoFrames) -> Optional[QualityReport]:
        """Step 8: Quality check."""
        print(f"[Step 8/9] Quality Control")
        print(f"  Analyzing frames...")
        
        tensor = frames.to_tensor()
        if tensor.dim() == 4:
            pass
        else:
            tensor = tensor.permute(1, 0, 2, 3)
        
        report, corrections = self.quality_controller.assess(tensor)
        
        print(f"  Quality Score: {report.overall_score:.2f}")
        print(f"  Issues: {[i.value for i in report.issues] if report.issues else 'None'}")
        
        if report.needs_correction and corrections:
            print(f"  Corrections applied: {list(corrections.keys())}")
        
        return report
    
    def _step9_export(self, frames: VideoFrames, output_path: Path) -> None:
        """Step 9: Export."""
        print(f"[Step 9/9] Export (FFmpeg)")
        print(f"  Format: {self.config.output_format}")
        print(f"  FPS: {self.config.fps}")
        print(f"  Quality: {self.config.quality}")
        print(f"  Output: {output_path}")
        
        quality_map = {
            "low": QualityPreset.LOW,
            "medium": QualityPreset.MEDIUM,
            "high": QualityPreset.HIGH,
            "ultra": QualityPreset.ULTRA
        }
        
        options = ExportOptions(
            video_spec=VideoSpec(
                duration_seconds=self.config.duration_seconds,
                fps=self.config.fps,
                format=VideoFormat(self.config.output_format),
                quality=quality_map.get(self.config.quality, QualityPreset.MEDIUM)
            ),
            enable_interpolation=self.config.enable_interpolation
        )
        
        self.exporter.export(frames, output_path, options)
        print(f"  Export complete!")
    
    def _get_default_prompt(self, content_type: str) -> str:
        """Get default prompt based on content type."""
        prompts = {
            "human": "smooth animation, natural motion, high quality",
            "furry": "animated furry character, smooth fur, natural motion",
            "animal": "animated animal, natural motion, high quality",
            "landscape": "cinematic landscape, wind movement, high quality",
            "scene": "cinematic scene, smooth animation, high quality",
            "object": "smooth animation, subtle motion, high quality"
        }
        return prompts.get(content_type, "animated scene, smooth motion, high quality")
    
    def clear_cache(self) -> None:
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_pipeline(
    image_path: Union[str, Path],
    prompt: str = "",
    config: Optional[Union[PipelineConfig, Dict[str, Any]]] = None,
    output_path: Optional[Union[str, Path]] = None
) -> PipelineResult:
    """
    Run the complete image-to-video pipeline.
    
    This is the main entry point for the Picture-Aliver system.
    
    Args:
        image_path: Path to input image file
        prompt: Optional text prompt for generation
        config: Pipeline configuration (dict or PipelineConfig)
        output_path: Optional output path override
        
    Returns:
        PipelineResult with output path and processing metadata
        
    Example:
        >>> result = run_pipeline(
        ...     image_path="photo.jpg",
        ...     prompt="cinematic animation",
        ...     config={"duration_seconds": 10, "fps": 24},
        ...     output_path="output.mp4"
        ... )
        >>> print(result.output_path)
    """
    if isinstance(config, dict):
        config = PipelineConfig(**config)
    elif config is None:
        config = PipelineConfig()
    
    pipeline = Pipeline(config)
    pipeline.initialize()
    
    return pipeline.run_pipeline(image_path, prompt, config, output_path)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Picture-Aliver: AI Image-to-Video Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py -i image.jpg -o video.mp4
  python main.py -i image.jpg -o video.mp4 --duration 10 --fps 24
  python main.py -i furry.png -o animation.mp4 --motion-prompt "gentle tail wag"
  python main.py -i image.jpg -o video.mp4 --quality high --interpolate
        """
    )
    
    parser.add_argument("-i", "--input", required=True, help="Input image path")
    parser.add_argument("-o", "--output", required=True, help="Output video path")
    parser.add_argument("-p", "--prompt", default="", help="Text prompt")
    parser.add_argument("--duration", type=float, default=3.0, help="Duration in seconds (5-120)")
    parser.add_argument("--fps", type=int, default=8, help="Frames per second")
    parser.add_argument("--width", type=int, default=512, help="Frame width")
    parser.add_argument("--height", type=int, default=512, help="Frame height")
    parser.add_argument("--steps", type=int, default=25, help="Inference steps")
    parser.add_argument("--scale", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--motion-strength", type=float, default=0.8, help="Motion strength")
    parser.add_argument("--motion-mode", default="auto", choices=["auto", "cinematic", "zoom", "pan", "subtle", "furry"])
    parser.add_argument("--motion-prompt", help="Natural language motion description")
    parser.add_argument("--quality", default="medium", choices=["low", "medium", "high", "ultra"])
    parser.add_argument("--format", default="mp4", choices=["mp4", "webm", "gif"])
    parser.add_argument("--interpolate", action="store_true", help="Enable frame interpolation")
    parser.add_argument("--no-stabilization", action="store_true", help="Disable stabilization")
    parser.add_argument("--no-quality-check", action="store_true", help="Skip quality check")
    parser.add_argument("--benchmark", action="store_true", help="Show GPU benchmarks")
    parser.add_argument("--device", choices=["cuda", "cpu"])
    
    args = parser.parse_args()
    
    if args.benchmark:
        from gpu_optimization import print_benchmark_table
        print_benchmark_table()
        return 0
    
    config = PipelineConfig(
        duration_seconds=max(5.0, min(120.0, args.duration)),
        fps=args.fps,
        width=args.width,
        height=args.height,
        guidance_scale=args.scale,
        num_inference_steps=args.steps,
        motion_strength=args.motion_strength,
        motion_mode=args.motion_mode,
        motion_prompt=args.motion_prompt,
        quality=args.quality,
        output_format=args.format,
        enable_stabilization=not args.no_stabilization,
        enable_interpolation=args.interpolate,
        enable_quality_check=not args.no_quality_check,
        device=args.device
    )
    
    try:
        result = run_pipeline(
            image_path=args.input,
            prompt=args.prompt,
            config=config,
            output_path=args.output
        )
        
        if result.success:
            print(f"\nSuccess! Video saved to: {result.output_path}")
            print(f"Processing time: {result.processing_time:.1f}s")
            return 0
        else:
            print(f"\nFailed: {result.errors}")
            return 1
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())