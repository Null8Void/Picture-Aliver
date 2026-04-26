"""
Picture-Aliver Extended Unified Model Interface

Provides unified interface for multiple video generation models:
- Wan 2.1 / Wan 2.2 (via Diffusers)
- LightX2V (lightweight inference framework)
- HunyuanVideo-I2V
- LTX-Video (Q8 quantized)
- CogVideo
- LongCat-Video
- Legacy pipeline

Usage:
    from src.picture_aliver.models import VideoModel, create_model
    
    model = create_model("wan21")
    result = model.generate(image="input.jpg", prompt="motion")
"""

from __future__ import annotations

import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import torch
import numpy as np
from PIL import Image

logger = logging.getLogger("picture_aliver.models")


class ModelType(Enum):
    """Supported model types."""
    WAN21 = "wan21"
    WAN22 = "wan22"
    LIGHTX2V = "lightx2v"
    HUNYUAN = "hunyuan"
    LTX = "ltx"
    COGVIDEO = "cogvideo"
    LONGCAT = "longcat"
    LEGACY = "legacy"


class GenerationStatus(Enum):
    """Generation status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class GenerationResult:
    """Result of video generation."""
    success: bool
    video_path: Optional[str] = None
    error: Optional[str] = None
    generation_time: Optional[float] = None
    status: GenerationStatus = GenerationStatus.PENDING
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass  
class ModelConfig:
    """Configuration for model."""
    model_type: ModelType = ModelType.WAN21
    model_id: str = ""
    num_frames: int = 81
    guidance_scale: float = 5.0
    num_inference_steps: int = 40
    fps: int = 16
    height: int = 480
    width: int = 832
    enable_offload: bool = True
    torch_dtype: str = "bfloat16"
    negative_prompt: str = ""
    output_dir: str = "./outputs"
    device: str = "cuda"
    # Model-specific options
    model_cls: str = ""
    task: str = "i2v"
    cpu_offload: bool = False
    quantization: Optional[str] = None
    use_distilled: bool = False


class VideoModel:
    """
    Unified video generation model interface.
    
    Supports multiple backends with a common API.
    """
    
    # Mapping of model types to their loader methods
    _LOADERS = {
        ModelType.WAN21: "_load_wan21",
        ModelType.WAN22: "_load_wan22",
        ModelType.LIGHTX2V: "_load_lightx2v",
        ModelType.HUNYUAN: "_load_hunyuan",
        ModelType.LTX: "_load_ltx",
        ModelType.COGVIDEO: "_load_cogvideo",
        ModelType.LONGCAT: "_load_longcat",
        ModelType.LEGACY: "_load_legacy",
    }
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self._pipeline = None
        self._device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._loaded = False
        self._model = None
        self._logger = logging.getLogger(f"picture_aliver.models.{config.model_type.value}")
        
    @property
    def is_loaded(self) -> bool:
        return self._loaded
    
    @property
    def device(self) -> str:
        return self._device
    
    def load(self) -> bool:
        """Load the model into memory."""
        if self._loaded:
            return True
            
        self._logger.info(f"Loading {self.config.model_type.value} model...")
        start_time = time.time()
        
        loader_method = self._LOADERS.get(self.config.model_type)
        if loader_method is None:
            self._logger.error(f"Unknown model type: {self.config.model_type}")
            return False
            
        try:
            success = getattr(self, loader_method)()
            self._loaded = success
            elapsed = time.time() - start_time
            self._logger.info(f"Model loaded in {elapsed:.1f}s")
            return success
            
        except Exception as e:
            self._logger.exception(f"Failed to load model: {e}")
            return False
    
    # =========================================================================
    # Model Loaders
    # =========================================================================
    
    def _load_wan21(self) -> bool:
        """Load Wan 2.1 model."""
        try:
            from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
            from diffusers.utils import load_image
            from transformers import CLIPVisionModel
            
            dtype = getattr(torch, self.config.torch_dtype, torch.bfloat16)
            self._logger.info(f"Loading Wan 2.1: {self.config.model_id}")
            
            self._pipeline = WanImageToVideoPipeline.from_pretrained(
                self.config.model_id,
                torch_dtype=dtype,
            )
            
            if self.config.enable_offload:
                self._pipeline.enable_model_cpu_offload()
            else:
                self._pipeline.to(self._device)
            
            self._pipeline_utils = {"load_image": load_image}
            self._logger.info("Wan 2.1 loaded")
            return True
            
        except Exception as e:
            self._logger.error(f"Wan 2.1 load failed: {e}")
            return False
    
    def _load_wan22(self) -> bool:
        """Load Wan 2.2 model (MoE architecture)."""
        try:
            from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
            from diffusers.utils import load_image
            
            dtype = getattr(torch, self.config.torch_dtype, torch.bfloat16)
            self._logger.info(f"Loading Wan 2.2: {self.config.model_id}")
            
            self._pipeline = WanImageToVideoPipeline.from_pretrained(
                self.config.model_id,
                torch_dtype=dtype,
            )
            
            if self.config.enable_offload:
                self._pipeline.enable_model_cpu_offload()
            
            self._pipeline_utils = {"load_image": load_image}
            self._logger.info("Wan 2.2 loaded")
            return True
            
        except Exception as e:
            self._logger.error(f"Wan 2.2 load failed: {e}")
            return False
    
    def _load_lightx2v(self) -> bool:
        """Load LightX2V framework."""
        try:
            from lightx2v import LightX2VPipeline
            
            self._logger.info(f"Loading LightX2V: {self.config.model_id}")
            
            # Determine model class from config
            model_cls = self.config.model_cls or "wan2.2_moe"
            
            self._pipeline = LightX2VPipeline(
                model_path=self.config.model_id,
                model_cls=model_cls,
                task=self.config.task,
            )
            
            if self.config.cpu_offload:
                self._pipeline.enable_offload(
                    cpu_offload=True,
                    offload_granularity="block",
                )
            
            self._logger.info("LightX2V loaded")
            return True
            
        except ImportError as e:
            self._logger.error(f"LightX2V not installed: {e}")
            return False
        except Exception as e:
            self._logger.error(f"LightX2V load failed: {e}")
            return False
    
    def _load_hunyuan(self) -> bool:
        """Load HunyuanVideo-I2V."""
        try:
            from diffusers import HunyuanVideoPipeline
            from diffusers.utils import load_image
            
            self._logger.info(f"Loading HunyuanVideo: {self.config.model_id}")
            
            self._pipeline = HunyuanVideoPipeline.from_pretrained(
                self.config.model_id,
                torch_dtype=torch.bfloat16,
            )
            
            if self.config.enable_offload:
                self._pipeline.enable_model_cpu_offload()
            
            self._pipeline_utils = {"load_image": load_image}
            self._logger.info("HunyuanVideo loaded")
            return True
            
        except ImportError as e:
            self._logger.error(f"HunyuanVideo not available: {e}")
            return False
        except Exception as e:
            self._logger.error(f"HunyuanVideo load failed: {e}")
            return False
    
    def _load_ltx(self) -> bool:
        """Load LTX-Video."""
        try:
            # Check for Q8 kernels
            try:
                from q8_kernels import quantize
                self._has_q8 = True
            except ImportError:
                self._has_q8 = False
            
            from diffusers import LTXVideoPipeline
            from diffusers.utils import load_image
            
            self._logger.info(f"Loading LTX-Video: {self.config.model_id}")
            
            self._pipeline = LTXVideoPipeline.from_pretrained(
                self.config.model_id,
                torch_dtype=torch.float16,  # Q8 needs float16
            )
            
            if self.config.enable_offload:
                self._pipeline.enable_model_cpu_offload()
            
            self._pipeline_utils = {"load_image": load_image}
            self._logger.info("LTX-Video loaded")
            return True
            
        except Exception as e:
            self._logger.error(f"LTX-Video load failed: {e}")
            return False
    
    def _load_cogvideo(self) -> bool:
        """Load CogVideo."""
        try:
            from diffusers import CogVideoPipeline
            from diffusers.utils import load_image
            
            self._logger.info(f"Loading CogVideo: {self.config.model_id}")
            
            self._pipeline = CogVideoPipeline.from_pretrained(
                self.config.model_id,
                torch_dtype=torch.bfloat16,
            )
            
            if self.config.enable_offload:
                self._pipeline.enable_model_cpu_offload()
            
            self._pipeline_utils = {"load_image": load_image}
            self._logger.info("CogVideo loaded")
            return True
            
        except Exception as e:
            self._logger.error(f"CogVideo load failed: {e}")
            return False
    
    def _load_longcat(self) -> bool:
        """Load LongCat-Video."""
        try:
            from diffusers import StableVideoDiffusionPipeline
            from diffusers.utils import load_image
            
            self._logger.info(f"Loading LongCat-Video")
            
            # LongCat uses SVD architecture
            self._pipeline = StableVideoDiffusionPipeline.from_pretrained(
                "stabilityai/stable-video-diffusion",
                torch_dtype=torch.bfloat16,
            )
            
            if self.config.enable_offload:
                self._pipeline.enable_model_cpu_offload()
            
            self._pipeline_utils = {"load_image": load_image}
            self._logger.info("LongCat-Video loaded")
            return True
            
        except Exception as e:
            self._logger.error(f"LongCat-Video load failed: {e}")
            return False
    
    def _load_legacy(self) -> bool:
        """Load legacy pipeline."""
        try:
            project_root = Path(__file__).parent.parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            
            from src.picture_aliver.main import Pipeline, PipelineConfig
            
            config = PipelineConfig(
                duration_seconds=self.config.num_frames / self.config.fps,
                fps=self.config.fps,
                width=self.config.width,
                height=self.config.height,
                guidance_scale=self.config.guidance_scale,
            )
            
            self._pipeline = Pipeline(config)
            self._pipeline.initialize()
            
            self._logger.info("Legacy pipeline loaded")
            return True
            
        except Exception as e:
            self._logger.error(f"Legacy load failed: {e}")
            return False
    
    # =========================================================================
    # Generation Methods
    # =========================================================================
    
    def generate(
        self,
        image: Union[str, Path, Image.Image],
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_frames: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        seed: int = -1,
        output_path: Optional[str] = None,
    ) -> GenerationResult:
        """Generate video from image."""
        if not self._loaded:
            if not self.load():
                return GenerationResult(
                    success=False,
                    error="Failed to load model",
                    status=GenerationStatus.FAILED,
                )
        
        start_time = time.time()
        negative_prompt = negative_prompt or self.config.negative_prompt
        num_frames = num_frames or self.config.num_frames
        guidance_scale = guidance_scale or self.config.guidance_scale
        num_inference_steps = num_inference_steps or self.config.num_inference_steps
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / f"video_{timestamp}.mp4")
        
        try:
            self._logger.info(f"Generating: {prompt}")
            
            if self.config.model_type in (ModelType.WAN21, ModelType.WAN22):
                result = self._generate_diffusers(
                    image, prompt, negative_prompt,
                    num_frames, guidance_scale, num_inference_steps,
                    output_path
                )
            elif self.config.model_type == ModelType.LIGHTX2V:
                result = self._generate_lightx2v(
                    image, prompt, negative_prompt,
                    num_frames, guidance_scale, num_inference_steps,
                    output_path
                )
            elif self.config.model_type == ModelType.LEGACY:
                result = self._generate_legacy(
                    image, prompt, output_path
                )
            else:
                # Fallback to diffusers interface
                result = self._generate_diffusers(
                    image, prompt, negative_prompt,
                    num_frames, guidance_scale, num_inference_steps,
                    output_path
                )
            
            result.generation_time = time.time() - start_time
            self._logger.info(f"Done in {result.generation_time:.1f}s")
            return result
            
        except Exception as e:
            self._logger.exception(f"Generation failed: {e}")
            return GenerationResult(
                success=False,
                error=str(e),
                status=GenerationStatus.FAILED,
            )
    
    def _generate_diffusers(
        self,
        image: Union[str, Path, Image.Image],
        prompt: str,
        negative_prompt: str,
        num_frames: int,
        guidance_scale: float,
        num_inference_steps: int,
        output_path: str,
    ) -> GenerationResult:
        """Generate using Diffusers pipeline."""
        from diffusers.utils import load_image, export_to_video
        
        # Load image
        if isinstance(image, (str, Path)):
            image = load_image(str(image))
        
        # Resize
        aspect_ratio = image.height / image.width
        max_area = self.config.height * self.config.width
        mod_value = 8
        
        height = int(np.sqrt(max_area * aspect_ratio) // mod_value * mod_value)
        width = int(np.sqrt(max_area / aspect_ratio) // mod_value * mod_value)
        image = image.resize((width, height))
        
        # Generate
        output = self._pipeline(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        ).frames[0]
        
        export_to_video(output, output_path, fps=self.config.fps)
        
        return GenerationResult(
            success=True,
            video_path=output_path,
            status=GenerationStatus.COMPLETED,
        )
    
    def _generate_lightx2v(
        self,
        image: Union[str, Path],
        prompt: str,
        negative_prompt: str,
        num_frames: int,
        guidance_scale: float,
        num_inference_steps: int,
        output_path: str,
    ) -> GenerationResult:
        """Generate using LightX2V."""
        image_path = str(image) if isinstance(image, (str, Path)) else None
        
        if image_path and not Path(image_path).exists():
            if isinstance(image, Image.Image):
                temp = Path(self.config.output_dir) / "temp.png"
                temp.parent.mkdir(parents=True, exist_ok=True)
                image.save(temp)
                image_path = str(temp)
        
        self._pipeline.generate(
            seed=-1,
            image_path=image_path,
            prompt=prompt,
            negative_prompt=negative_prompt,
            save_result_path=output_path,
        )
        
        return GenerationResult(
            success=True,
            video_path=output_path,
            status=GenerationStatus.COMPLETED,
        )
    
    def _generate_legacy(
        self,
        image: Union[str, Path],
        prompt: str,
        output_path: str,
    ) -> GenerationResult:
        """Generate using legacy pipeline."""
        from src.picture_aliver.main import Pipeline, PipelineConfig
        
        duration = self.config.num_frames / self.config.fps
        
        result = self._pipeline.run_pipeline(
            image_path=str(image),
            prompt=prompt,
            output_path=output_path,
        )
        
        return GenerationResult(
            success=result.success,
            video_path=output_path if result.success else None,
            error=", ".join(result.errors) if result.errors else None,
            status=GenerationStatus.COMPLETED if result.success else GenerationStatus.FAILED,
        )
    
    def unload(self):
        """Unload model from memory."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._loaded = False
        self._logger.info("Model unloaded")
    
    def get_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_type": self.config.model_type.value,
            "loaded": self._loaded,
            "device": self._device,
            "config": {
                "model_id": self.config.model_id,
                "num_frames": self.config.num_frames,
                "guidance_scale": self.config.guidance_scale,
                "num_inference_steps": self.config.num_inference_steps,
            }
        }
    
    def __del__(self):
        self.unload()


# =============================================================================
# Factory Functions
# =============================================================================

def create_model(
    model_type: Union[str, ModelType] = "wan21",
    config: Optional[Dict[str, Any]] = None,
) -> VideoModel:
    """Create a VideoModel instance."""
    if isinstance(model_type, str):
        model_type = ModelType(model_type.lower())
    
    config = config or {}
    model_config = ModelConfig(model_type=model_type, **config)
    
    return VideoModel(model_config)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load model configuration from YAML."""
    import yaml
    
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "configs" / "model_config_extended.yaml"
    
    config_path = Path(config_path)
    if not config_path.exists():
        return {}
    
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_model_from_config(config_path: Optional[str] = None) -> VideoModel:
    """Create VideoModel from config file."""
    config = load_config(config_path)
    
    model_type = config.pop("primary_model", "wan21")
    common = config.pop("common", {})
    
    model_config = config.get(model_type, {})
    model_config = {**common, **model_config}
    
    return create_model(model_type, model_config)


# =============================================================================
# Validation
# =============================================================================

def validate_model(model_type: str = "wan21") -> Dict[str, Any]:
    """Validate model installation."""
    results = {
        "model_type": model_type,
        "available": False,
        "errors": [],
        "warnings": [],
    }
    
    # Check Python
    if sys.version_info < (3, 9):
        results["errors"].append(f"Python 3.9+ required")
    
    # Check base dependencies
    for name, import_name in [("torch", "torch"), ("numpy", "numpy")]:
        try:
            __import__(import_name)
        except ImportError:
            results["errors"].append(f"Missing: {name}")
    
    # Check model-specific
    try:
        if model_type in ("wan21", "wan22"):
            from diffusers import WanImageToVideoPipeline
            results["available"] = True
        elif model_type == "lightx2v":
            from lightx2v import LightX2VPipeline
            results["available"] = True
        elif model_type == "hunyuan":
            from diffusers import HunyuanVideoPipeline
            results["available"] = True
        elif model_type == "ltx":
            from diffusers import LTXVideoPipeline
            results["available"] = True
        elif model_type in ("cogvideo", "longcat"):
            results["available"] = True  # Uses SVD
        elif model_type == "legacy":
            results["available"] = True
    except ImportError as e:
        results["warnings"].append(f"Install: pip install {model_type}")
    
    # GPU info
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        results["vram_gb"] = round(props.total_memory / (1024**3), 1)
        results["gpu_name"] = props.name
    else:
        results["warnings"].append("No GPU - will use CPU")
    
    results["passed"] = len(results["errors"]) == 0
    return results


def validate_all_models() -> Dict[str, Dict]:
    """Validate all available models."""
    model_types = ["wan21", "wan22", "lightx2v", "hunyuan", "ltx", "legacy"]
    return {m: validate_model(m) for m in model_types}


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Picture-Aliver Models")
    parser.add_argument("--validate", action="store_true", help="Validate all models")
    parser.add_argument("--model", default="wan21", help="Model to validate")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s"
    )
    
    if args.validate:
        print("\n=== Model Validation ===")
        results = validate_all_models()
        for name, result in results.items():
            status = "OK" if result["passed"] else "FAIL"
            print(f"\n[{status}] {name}")
            print(f"  Available: {result.get('available', False)}")
            if result.get("warnings"):
                for w in result["warnings"]:
                    print(f"  Warning: {w}")
    else:
        result = validate_model(args.model)
        print(f"\nValidation: {args.model}")
        print(f"Passed: {result['passed']}")
        print(f"Available: {result.get('available', False)}")