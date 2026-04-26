#!/usr/bin/env python3
"""
Picture-Aliver CLI - Command-line interface for video generation

Usage:
    python -m src.picture_aliver.cli generate --image input.jpg --prompt "wave"
    python -m src.picture_aliver.cli list-models
    python -m src.picture_aliver.cli status
"""

import argparse
import sys
import logging
from pathlib import Path


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )


def cmd_generate(args):
    """Generate video from image."""
    from .model_manager_extended import get_manager
    
    manager = get_manager()
    
    print(f"\nGenerating video from: {args.image}")
    print(f"Prompt: {args.prompt}")
    print(f"Model order: {manager.get_status()['model_order']}")
    
    result = manager.generate(
        image=args.image,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        duration=args.duration,
        fps=args.fps,
        width=args.width,
        height=args.height,
    )
    
    if result["success"]:
        print(f"\n[OK] Success!")
        print(f"  Video: {result['video_path']}")
        print(f"  Model: {result['model_type']}")
        print(f"  Time: {result['generation_time']:.1f}s")
    
    else:
        print(f"\n[FAIL] Failed: {result.get('error')}")
        print(f"  Attempts:")
        for attempt in result.get("attempts", []):
            status = "OK" if attempt["success"] else "FAIL"
            print(f"    [{status}] {attempt['model']}: {attempt.get('error', 'OK')}")
        
        sys.exit(1)


def cmd_list_models(args):
    """List available models."""
    from .model_manager_extended import get_manager
    
    manager = get_manager()
    status = manager.get_status()
    
    print("\n=== Available Models ===\n")
    print(f"GPU: {status['gpu_name'] or 'None (CPU mode)'}")
    print(f"VRAM: {status['vram_gb']:.1f}GB\n")
    
    from .model_manager_extended import MODEL_INFO
    
    # Use ASCII-safe characters
    for model_type, info in MODEL_INFO.items():
        available = model_type in status["model_order"]
        indicator = "x" if available else " "
        
        print(f"[{indicator}] {info.display_name} ({model_type})")
        print(f"    License: {info.license}")
        print(f"    VRAM: {info.min_vram_gb}GB | Quality: {info.quality} | Speed: {info.speed}")
        print()


def cmd_status(args):
    """Show system status."""
    from .model_manager_extended import get_manager
    from .models_extended import validate_all_models
    
    manager = get_manager()
    status = manager.get_status()
    
    print("\n=== System Status ===\n")
    print(f"Primary Model: {status['primary']}")
    print(f"Fallback Models: {status['fallback']}")
    print(f"Model Order: {status['model_order']}")
    print()
    print(f"GPU Available: {status['gpu_available']}")
    print(f"GPU Name: {status['gpu_name'] or 'None'}")
    print(f"VRAM: {status['vram_gb']:.1f}GB")
    print()
    
    if args.validate:
        print("=== Model Validation ===\n")
        results = validate_all_models()
        for name, result in results.items():
            status_icon = "✓" if result.get("available") else "✗"
            print(f"[{status_icon}] {name}")
            if result.get("warnings"):
                for w in result["warnings"]:
                    print(f"    Warning: {w}")
            print()


def cmd_validate(args):
    """Validate specific model."""
    from .models_extended import validate_model
    
    result = validate_model(args.model)
    
    print(f"\n=== Validation: {args.model} ===\n")
    print(f"Available: {result.get('available', False)}")
    print(f"Passed: {result['passed']}")
    
    if result.get("errors"):
        print("\nErrors:")
        for e in result["errors"]:
            print(f"  ✗ {e}")
    
    if result.get("warnings"):
        print("\nWarnings:")
        for w in result["warnings"]:
            print(f"  ! {w}")
    
    if result.get("gpu_name"):
        print(f"\nGPU: {result['gpu_name']}")
        print(f"VRAM: {result.get('vram_gb', '?')}GB")
    
    sys.exit(0 if result["passed"] else 1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Picture-Aliver CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--log-level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # generate command
    gen_parser = subparsers.add_parser("generate", help="Generate video")
    gen_parser.add_argument("--image", "-i", required=True, help="Input image path")
    gen_parser.add_argument("--prompt", "-p", required=True, help="Animation prompt")
    gen_parser.add_argument("--negative-prompt", "-n", default="", help="Negative prompt")
    gen_parser.add_argument("--duration", "-d", type=float, default=3.0, help="Duration (seconds)")
    gen_parser.add_argument("--fps", type=int, default=8, help="Frames per second")
    gen_parser.add_argument("--width", type=int, default=512, help="Output width")
    gen_parser.add_argument("--height", type=int, default=512, help="Output height")
    gen_parser.add_argument("-o", "--output", help="Output path")
    
    # list-models command
    subparsers.add_parser("list-models", help="List available models")
    
    # status command
    status_parser = subparsers.add_parser("status", help="Show system status")
    status_parser.add_argument("--validate", action="store_true", help="Validate models")
    
    # validate command
    val_parser = subparsers.add_parser("validate", help="Validate a model")
    val_parser.add_argument("model", help="Model type to validate")
    
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    
    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "list-models":
        cmd_list_models(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "validate":
        cmd_validate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()