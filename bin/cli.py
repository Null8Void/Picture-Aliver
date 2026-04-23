#!/usr/bin/env python
"""Command-line interface for Image2Video AI system."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch

from src.core.pipeline import Image2VideoPipeline, PipelineConfig
from src.core.device import get_torch_device
from src.utils.logger import setup_logger


def create_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="Image2Video",
        description="Convert a single image into a coherent animated video using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input photo.jpg --output video.mp4
  %(prog)s -i photo.jpg -o video.mp4 --prompt "slow ocean waves" --num-frames 48
  %(prog)s -i photo.jpg --motion-mode camera --fps 12
  %(prog)s -i photo.jpg --no-depth --no-segmentation
        """
    )
    
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Input image path"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output video path (default: input_stem_video.mp4)"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Text prompt for video generation"
    )
    
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="blurry, low quality, artifacts, static",
        help="Negative prompt"
    )
    
    parser.add_argument(
        "--num-frames",
        type=int,
        default=24,
        help="Number of frames to generate (default: 24)"
    )
    
    parser.add_argument(
        "--fps",
        type=int,
        default=8,
        help="Frames per second (default: 8)"
    )
    
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="CFG guidance scale (default: 7.5)"
    )
    
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=25,
        help="Number of denoising steps (default: 25)"
    )
    
    parser.add_argument(
        "--motion-strength",
        type=float,
        default=0.8,
        help="Motion strength 0-1 (default: 0.8)"
    )
    
    parser.add_argument(
        "--motion-mode",
        type=str,
        choices=["auto", "camera", "flow", "keyframes", "zoom", "pan"],
        default="auto",
        help="Motion mode (default: auto)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu", "mps"],
        help="Compute device"
    )
    
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Output resolution (default: 512)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--no-depth",
        action="store_true",
        help="Disable depth estimation"
    )
    
    parser.add_argument(
        "--no-segmentation",
        action="store_true",
        help="Disable segmentation"
    )
    
    parser.add_argument(
        "--no-consistency",
        action="store_true",
        help="Disable temporal consistency"
    )
    
    parser.add_argument(
        "--no-3d-effects",
        action="store_true",
        help="Disable depth-based effects"
    )
    
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Save intermediate results (depth, segmentation, flow)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="Image2Video AI v1.0.0"
    )
    
    return parser


def main(args: Optional[list] = None) -> int:
    """Main CLI entry point."""
    parser = create_parser()
    parsed_args = parser.parse_args(args)
    
    logger = setup_logger(
        name="Image2Video",
        level=10 if parsed_args.verbose else 20
    )
    
    input_path = Path(parsed_args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1
    
    if parsed_args.output:
        output_path = Path(parsed_args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_video.mp4"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    device = get_torch_device(parsed_args.device)
    
    config = PipelineConfig(
        enable_depth=not parsed_args.no_depth,
        enable_segmentation=not parsed_args.no_segmentation,
        enable_motion=True,
        enable_consistency=not parsed_args.no_consistency,
        use_3d_effects=not parsed_args.no_3d_effects,
        motion_mode=parsed_args.motion_mode,
        verbose=True
    )
    
    pipeline = Image2VideoPipeline(config=config, device=device)
    
    try:
        logger.info("Starting Image2Video processing...")
        logger.info(f"Input: {input_path}")
        logger.info(f"Output: {output_path}")
        logger.info(f"Device: {device}")
        logger.info(f"Frames: {parsed_args.num_frames}, FPS: {parsed_args.fps}")
        
        result = pipeline.process(
            image=str(input_path),
            prompt=parsed_args.prompt,
            negative_prompt=parsed_args.negative_prompt,
            num_frames=parsed_args.num_frames,
            fps=parsed_args.fps,
            guidance_scale=parsed_args.guidance_scale,
            num_inference_steps=parsed_args.num_inference_steps,
            motion_strength=parsed_args.motion_strength,
            output_path=str(output_path),
            seed=parsed_args.seed
        )
        
        logger.info(f"Video saved to: {output_path}")
        logger.info(f"Duration: {result.duration:.2f}s, Processing time: {result.processing_time:.2f}s")
        
        if parsed_args.save_intermediate:
            intermediate_dir = output_path.parent / f"{output_path.stem}_intermediate"
            pipeline.save_intermediate(result, intermediate_dir)
            logger.info(f"Intermediate results saved to: {intermediate_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        if parsed_args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    finally:
        pipeline.clear_cache()


if __name__ == "__main__":
    sys.exit(main())