"""Semantic segmentation using Segment Anything Model (SAM) and other models."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from .types import Mask, ObjectDetection, SegmentationMask


class Segmentor:
    """Unified semantic segmentation interface supporting multiple backends.
    
    Supports:
    - SAM (Segment Anything Model): Point/mask-based segmentation
    - DeepLabV3: Semantic segmentation with class labels
    - Custom: Lightweight fallback model
    
    Args:
        config: Segmentation configuration object
        device: Target compute device
    """
    
    SUPPORTED_MODELS = ["sam", "deeplabv3", "custom"]
    
    def __init__(
        self,
        config: Optional[object] = None,
        device: Optional[torch.device] = None
    ):
        self.config = config or self._default_config()
        self.device = device or torch.device("cpu")
        self.model = None
        self.processor = None
        self.predictor = None
        self.model_type = getattr(self.config, 'name', 'SAM').lower()
        self._initialized = False
    
    def _default_config(self):
        """Get default configuration."""
        class DefaultConfig:
            name = "SAM"
            model_type = "vit_h"
            checkpoint = ""
            device = "cuda"
        return DefaultConfig()
    
    def initialize(self) -> None:
        """Initialize the segmentation model."""
        if self._initialized:
            return
        
        if self.model_type == "sam":
            self._init_sam()
        elif self.model_type == "deeplabv3":
            self._init_deeplabv3()
        else:
            self._init_sam()
        
        self._initialized = True
    
    def _init_sam(self) -> None:
        """Initialize Segment Anything Model."""
        try:
            from segment_anything import sam_model_registry, SamPredictor
            
            model_type = getattr(self.config, 'model_type', 'vit_h')
            checkpoint = getattr(self.config, 'checkpoint', None)
            cache_dir = getattr(self.config, 'cache_dir', 'models/segmentation')
            
            if checkpoint and not checkpoint.startswith("http"):
                if not torch.path.exists(checkpoint):
                    checkpoint = None
            
            if checkpoint is None:
                model_urls = {
                    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                }
            
            try:
                self.model = sam_model_registry[model_type](checkpoint=checkpoint)
            except Exception:
                import urllib.request
                import os
                
                url = model_urls.get(model_type, model_urls["vit_h"])
                save_path = os.path.join(cache_dir, f"sam_{model_type}.pth")
                
                if not os.path.exists(save_path):
                    print(f"Downloading SAM model from {url}...")
                    os.makedirs(cache_dir, exist_ok=True)
                    urllib.request.urlretrieve(url, save_path)
                
                self.model = sam_model_registry[model_type](checkpoint=save_path)
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.predictor = SamPredictor(self.model)
            
        except ImportError as e:
            print(f"SAM not available: {e}. Using fallback segmentation.")
            self.model = self._create_fallback_segmentor()
    
    def _init_deeplabv3(self) -> None:
        """Initialize DeepLabV3 model."""
        try:
            from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
            
            model_name = "Intel/deeplabv3-large"
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            
        except ImportError:
            self.model = self._create_fallback_segmentor()
    
    def _create_fallback_segmentor(self) -> nn.Module:
        """Create a lightweight fallback segmentation model."""
        class SimpleSegNet(nn.Module):
            def __init__(self, num_classes: int = 20):
                super().__init__()
                self.num_classes = num_classes
                
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                )
                
                self.decoder = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                    nn.Conv2d(128, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                    nn.Conv2d(64, 32, 3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, num_classes, 1),
                )
            
            def forward(self, x):
                return self.decoder(self.encoder(x))
        
            def predict(self, image):
                with torch.no_grad():
                    img_tensor = self.preprocess_image(image)
                    logits = self.forward(img_tensor)
                    pred = logits.argmax(dim=1)
                    return pred.squeeze(0)
            
            @staticmethod
            def preprocess_image(image):
                if isinstance(image, Image.Image):
                    image = np.array(image)
                if image.ndim == 2:
                    image = np.stack([image] * 3, axis=-1)
                if image.max() > 1:
                    image = image.astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
                return img_tensor.unsqueeze(0)
        
        model = SimpleSegNet()
        model = model.to(self.device)
        model.eval()
        return model
    
    def segment(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        points: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        masks: Optional[List[np.ndarray]] = None,
        return_logits: bool = False
    ) -> SegmentationMask:
        """Perform segmentation on an input image.
        
        Args:
            image: Input image
            points: Optional point prompts [N, 2] (x, y) in pixel coordinates
            labels: Point labels [N] (1=foreground, 0=background)
            masks: Optional mask prompts
            return_logits: Whether to return raw logits
            
        Returns:
            SegmentationMask object containing all detected masks
        """
        if not self._initialized:
            self.initialize()
        
        original_size = None
        if isinstance(image, Image.Image):
            image = np.array(image)
            original_size = image.shape[:2]
        elif isinstance(image, np.ndarray):
            original_size = image.shape[:2]
        
        if self.model_type == "sam":
            return self._segment_sam(image, points, labels, masks)
        elif self.model_type == "deeplabv3":
            return self._segment_deeplabv3(image)
        else:
            return self._segment_fallback(image)
    
    def _segment_sam(
        self,
        image: np.ndarray,
        points: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        masks: Optional[List[np.ndarray]] = None
    ) -> SegmentationMask:
        """Segment using SAM."""
        if image.max() <= 1:
            image = (image * 255).astype(np.uint8)
        elif image.dtype != np.uint8:
            image = image.astype(np.uint8)
        
        self.predictor.set_image(image)
        
        mask_list = []
        
        if points is not None and len(points) > 0:
            if labels is None:
                labels = np.ones(len(points), dtype=np.int32)
            
            mask_input, scores, masks = self.predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=True
            )
            
            for i, mask in enumerate(masks):
                mask_list.append(Mask(
                    segmentation=mask,
                    label="sam_object",
                    confidence=float(scores[i]) if i < len(scores) else None,
                    id=i
                ))
        
        if masks is not None:
            for i, mask in enumerate(masks):
                mask_list.append(Mask(
                    segmentation=mask,
                    label="prompted",
                    id=len(mask_list)
                ))
        
        if not mask_list:
            points, labels = self._generate_grid_points(image.shape[:2])
            mask_input, scores, predicted_masks = self.predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=True
            )
            
            for i, mask in enumerate(predicted_masks):
                mask_list.append(Mask(
                    segmentation=mask,
                    label="grid_object",
                    confidence=float(scores[i]) if i < len(scores) else None,
                    id=i
                ))
        
        return SegmentationMask(
            image_size=image.shape[:2],
            masks=mask_list,
            scores=np.array([m.confidence or 0.0 for m in mask_list])
        )
    
    def _generate_grid_points(
        self,
        image_size: Tuple[int, int],
        grid_size: int = 8
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate grid of points for automatic segmentation."""
        h, w = image_size
        y_coords = np.linspace(h * 0.1, h * 0.9, grid_size, dtype=np.int32)
        x_coords = np.linspace(w * 0.1, w * 0.9, grid_size, dtype=np.int32)
        
        points = []
        labels = []
        
        for y in y_coords:
            for x in x_coords:
                points.append([x, y])
                labels.append(1)
        
        return np.array(points), np.array(labels)
    
    def _segment_deeplabv3(self, image: np.ndarray) -> SegmentationMask:
        """Segment using DeepLabV3."""
        if isinstance(image, torch.Tensor):
            image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        if image.max() <= 1:
            image = (image * 255).astype(np.uint8)
        
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        pred = logits.argmax(dim=1).squeeze(0)
        
        masks = []
        unique_classes = pred.unique()
        
        for i, class_id in enumerate(unique_classes):
            if class_id == self.processor.model_config.num_labels - 1:
                continue
            
            mask = (pred == class_id).cpu().numpy()
            
            masks.append(Mask(
                segmentation=mask,
                label=str(class_id.item()),
                id=i
            ))
        
        return SegmentationMask(
            image_size=image.shape[:2],
            masks=masks,
            semantics=pred.cpu(),
            class_labels={i: str(i) for i in range(self.processor.model_config.num_labels)}
        )
    
    def _segment_fallback(self, image: np.ndarray) -> SegmentationMask:
        """Segment using fallback model."""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if image.max() <= 1:
            image = (image * 255).astype(np.uint8)
        
        img_tensor = self._preprocess_image(image)
        img_tensor = img_tensor.to(self.device)
        
        with torch.no_grad():
            pred = self.model.predict(img_tensor)
        
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        
        masks = []
        unique_classes = np.unique(pred)
        
        class_mapping = {
            0: "background", 1: "person", 2: "vehicle", 3: "animal",
            4: "furniture", 5: "plant", 6: "building", 7: "sky",
            8: "ground", 9: "object", 10: "text", 11: "food",
            12: "drink", 13: "clothing", 14: "electronics", 15: "art",
            16: "nature", 17: "sport", 18: "vehicle", 19: "other"
        }
        
        for i, class_id in enumerate(unique_classes):
            if class_id == 0:
                continue
            
            mask = (pred == class_id).astype(np.uint8)
            masks.append(Mask(
                segmentation=mask,
                label=class_mapping.get(int(class_id), f"class_{class_id}"),
                id=i
            ))
        
        return SegmentationMask(
            image_size=image.shape[:2],
            masks=masks,
            semantics=pred,
            class_labels=class_mapping
        )
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for segmentation."""
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        if image.shape[-1] == 4:
            image = image[:, :, :3]
        
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        return img_tensor.unsqueeze(0)
    
    def segment_with_prompts(
        self,
        image: Union[Image.Image, np.ndarray],
        prompt_type: str = "automatic",
        num_points: int = 16,
        points_per_side: int = 32
    ) -> SegmentationMask:
        """Segment with automatic or manual prompts.
        
        Args:
            image: Input image
            prompt_type: Type of prompts ('automatic', 'grid', 'center')
            num_points: Number of points for grid/center prompts
            points_per_side: Points per side for automatic segmentation
            
        Returns:
            SegmentationMask with generated masks
        """
        if not self._initialized:
            self.initialize()
        
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if self.model_type == "sam":
            try:
                from segment_anything import SamAutomaticMaskGenerator
                
                generator = SamAutomaticMaskGenerator(
                    model=self.model,
                    points_per_side=points_per_side,
                    pred_iou_thresh=getattr(self.config, 'pred_iou_thresh', 0.86),
                    stability_score_thresh=getattr(self.config, 'stability_score_thresh', 0.92),
                    crop_n_layers=0,
                    crop_n_points_downscale_factor=1,
                )
                
                result = generator.generate(image)
                
                masks = []
                for i, r in enumerate(result):
                    masks.append(Mask(
                        segmentation=r["segmentation"],
                        area=r.get("area"),
                        bbox=r.get("bbox"),
                        label="sam_auto",
                        confidence=r.get("predicted_iou"),
                        id=i
                    ))
                
                return SegmentationMask(
                    image_size=image.shape[:2],
                    masks=masks,
                    scores=np.array([m.confidence for m in masks])
                )
                
            except Exception:
                pass
        
        points, labels = self._generate_grid_points(image.shape[:2], num_points)
        return self.segment(image, points, labels)
    
    def segment_instances(self, image: Union[Image.Image, np.ndarray]) -> ObjectDetection:
        """Segment instances (bounding boxes + masks).
        
        Args:
            image: Input image
            
        Returns:
            ObjectDetection with boxes, labels, scores, and masks
        """
        seg_result = self.segment_with_prompts(image)
        
        boxes = []
        labels = []
        scores = []
        masks = []
        
        for mask in seg_result.masks:
            if mask.bbox is not None:
                x, y, w, h = mask.bbox
                boxes.append([x, y, x + w, y + h])
                labels.append(mask.label or "object")
                scores.append(mask.confidence or 0.5)
                masks.append(mask.segmentation)
        
        return ObjectDetection(
            boxes=np.array(boxes) if boxes else np.zeros((0, 4)),
            labels=labels,
            scores=np.array(scores) if scores else np.zeros(0),
            masks=np.array(masks) if masks else None
        )
    
    def __repr__(self) -> str:
        return f"Segmentor(model={self.model_type}, device={self.device})"