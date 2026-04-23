"""Type definitions for segmentation module."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch


@dataclass
class Mask:
    """Individual segmentation mask.
    
    Attributes:
        segmentation: Binary mask as boolean array or tensor
        area: Area of the mask in pixels
        bbox: Bounding box [x, y, width, height]
        label: Semantic label or class name
        confidence: Confidence score for the segmentation
        id: Unique identifier for the mask
    """
    segmentation: Union[np.ndarray, torch.Tensor]
    area: Optional[int] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    label: Optional[str] = None
    confidence: Optional[float] = None
    id: Optional[int] = None
    
    def __post_init__(self):
        if self.area is None:
            self._compute_area()
        if self.bbox is None:
            self._compute_bbox()
    
    def _compute_area(self) -> None:
        """Compute mask area."""
        if isinstance(self.segmentation, np.ndarray):
            self.area = int(self.segmentation.sum())
        else:
            self.area = int(self.segmentation.sum().item())
    
    def _compute_bbox(self) -> Tuple[int, int, int, int]:
        """Compute bounding box from mask."""
        if isinstance(self.segmentation, torch.Tensor):
            mask = self.segmentation.cpu().numpy()
        else:
            mask = self.segmentation
        
        if mask.ndim == 3:
            mask = mask.any(axis=-1) if mask.shape[-1] > 1 else mask.squeeze(-1)
        
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        self.bbox = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
        return self.bbox
    
    def to(self, device: torch.device) -> "Mask":
        """Move mask to device."""
        if isinstance(self.segmentation, torch.Tensor):
            return Mask(
                segmentation=self.segmentation.to(device),
                area=self.area,
                bbox=self.bbox,
                label=self.label,
                confidence=self.confidence,
                id=self.id
            )
        return self
    
    def numpy(self) -> "Mask":
        """Convert to numpy."""
        return Mask(
            segmentation=self.segmentation if isinstance(self.segmentation, np.ndarray) 
                        else self.segmentation.cpu().numpy(),
            area=self.area,
            bbox=self.bbox,
            label=self.label,
            confidence=self.confidence,
            id=self.id
        )
    
    def get_center(self) -> Tuple[float, float]:
        """Get center point of mask."""
        if isinstance(self.segmentation, torch.Tensor):
            mask = self.segmentation.cpu().numpy()
        else:
            mask = self.segmentation
        
        if mask.ndim == 3:
            mask = mask.squeeze(-1)
        
        y_indices, x_indices = np.where(mask)
        if len(y_indices) == 0:
            return (0.0, 0.0)
        
        return (float(np.mean(x_indices)), float(np.mean(y_indices)))
    
    def get_boundary(self) -> np.ndarray:
        """Get boundary pixels of the mask."""
        import scipy.ndimage as ndimage
        
        if isinstance(self.segmentation, torch.Tensor):
            mask = self.segmentation.cpu().numpy()
        else:
            mask = self.segmentation
        
        if mask.ndim == 3:
            mask = mask.any(axis=-1).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8) if mask.dtype != np.uint8 else mask
        
        eroded = ndimage.binary_erosion(mask)
        boundary = mask & ~eroded
        return boundary.astype(np.uint8)
    
    def get_convex_hull(self) -> np.ndarray:
        """Get convex hull of the mask."""
        from scipy.spatial import ConvexHull
        from scipy.ndimage import binary_fill_holes
        
        if isinstance(self.segmentation, torch.Tensor):
            mask = self.segmentation.cpu().numpy()
        else:
            mask = self.segmentation
        
        if mask.ndim == 3:
            mask = mask.any(axis=-1)
        
        y_indices, x_indices = np.where(mask)
        if len(y_indices) < 3:
            return mask
        
        points = np.column_stack([x_indices, y_indices])
        
        try:
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            
            hull_mask = np.zeros_like(mask)
            hull_mask[hull_points[:, 1].astype(int), hull_points[:, 0].astype(int)] = 1
            hull_mask = binary_fill_holes(hull_mask)
            
            return hull_mask
        except Exception:
            return mask.astype(np.uint8)
    
    def iou(self, other: "Mask") -> float:
        """Compute IoU with another mask."""
        if isinstance(self.segmentation, np.ndarray):
            mask1 = self.segmentation.astype(bool)
            mask2 = other.segmentation.astype(bool) if isinstance(other.segmentation, np.ndarray) else other.segmentation.cpu().numpy().astype(bool)
        else:
            mask1 = self.segmentation.cpu().numpy().astype(bool)
            mask2 = other.segmentation if isinstance(other.segmentation, np.ndarray) else other.segmentation.cpu().numpy().astype(bool)
        
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        
        if union == 0:
            return 0.0
        return float(intersection / union)


@dataclass
class ObjectDetection:
    """Object detection result.
    
    Attributes:
        boxes: Bounding boxes [N, 4] (x1, y1, x2, y2)
        labels: Class labels [N]
        scores: Confidence scores [N]
        masks: Optional segmentation masks [N, H, W]
    """
    boxes: Union[np.ndarray, torch.Tensor]
    labels: Union[np.ndarray, torch.Tensor, List[str]]
    scores: Union[np.ndarray, torch.Tensor]
    masks: Optional[Union[np.ndarray, torch.Tensor]] = None
    
    def __post_init__(self):
        if isinstance(self.boxes, torch.Tensor):
            self.boxes = self.boxes.cpu().numpy()
        if isinstance(self.labels, torch.Tensor):
            self.labels = self.labels.cpu().numpy()
        if isinstance(self.scores, torch.Tensor):
            self.scores = self.scores.cpu().numpy()
        if self.masks is not None and isinstance(self.masks, torch.Tensor):
            self.masks = self.masks.cpu().numpy()
    
    def __len__(self) -> int:
        return len(self.boxes)
    
    def __getitem__(self, idx: int) -> Tuple:
        return (
            self.boxes[idx],
            self.labels[idx],
            self.scores[idx],
            self.masks[idx] if self.masks is not None else None
        )
    
    def filter(self, min_score: float = 0.5) -> "ObjectDetection":
        """Filter detections by minimum score."""
        mask = self.scores >= min_score
        return ObjectDetection(
            boxes=self.boxes[mask],
            labels=self.labels[mask],
            scores=self.scores[mask],
            masks=self.masks[mask] if self.masks is not None else None
        )


@dataclass
class SegmentationMask:
    """Container for segmentation output.
    
    Attributes:
        image_size: Original image dimensions (H, W)
        masks: List of individual masks
        class_labels: Class label mapping
        scores: Per-mask confidence scores
        semantics: Semantic segmentation map (class per pixel)
        panoptic: Panoptic segmentation combining instances
    """
    image_size: Tuple[int, int]
    masks: List[Mask] = field(default_factory=list)
    class_labels: Optional[Dict[int, str]] = None
    scores: Optional[Union[np.ndarray, torch.Tensor]] = None
    semantics: Optional[Union[np.ndarray, torch.Tensor]] = None
    panoptic: Optional[Union[np.ndarray, torch.Tensor]] = None
    
    def __len__(self) -> int:
        return len(self.masks)
    
    def __getitem__(self, idx: int) -> Mask:
        return self.masks[idx]
    
    @property
    def combined_mask(self) -> Union[np.ndarray, torch.Tensor]:
        """Combine all masks into a single mask."""
        if not self.masks:
            return np.zeros(self.image_size, dtype=np.uint8)
        
        combined = np.zeros(self.image_size, dtype=np.uint8)
        for i, mask in enumerate(self.masks, 1):
            if isinstance(mask.segmentation, torch.Tensor):
                seg = mask.segmentation.cpu().numpy()
            else:
                seg = mask.segmentation
            combined[seg] = i
        
        return combined
    
    def get_foreground_mask(self) -> Union[np.ndarray, torch.Tensor]:
        """Get binary foreground mask (all objects)."""
        combined = self.combined_mask
        return (combined > 0).astype(np.uint8)
    
    def get_by_label(self, label: str) -> List[Mask]:
        """Get all masks with a specific label."""
        if self.class_labels is None:
            return []
        
        label_id = None
        for lid, lname in self.class_labels.items():
            if lname == label:
                label_id = lid
                break
        
        if label_id is None:
            return []
        
        return [m for m in self.masks if getattr(m, 'label_id', None) == label_id]
    
    def to(self, device: torch.device) -> "SegmentationMask":
        """Move all masks to device."""
        return SegmentationMask(
            image_size=self.image_size,
            masks=[m.to(device) for m in self.masks],
            class_labels=self.class_labels,
            scores=self.scores.to(device) if self.scores is not None and isinstance(self.scores, torch.Tensor) else self.scores,
            semantics=self.semantics.to(device) if self.semantics is not None and isinstance(self.semantics, torch.Tensor) else self.semantics,
            panoptic=self.panoptic.to(device) if self.panoptic is not None and isinstance(self.panoptic, torch.Tensor) else self.panoptic
        )
    
    def numpy(self) -> "SegmentationMask":
        """Convert all tensors to numpy arrays."""
        return SegmentationMask(
            image_size=self.image_size,
            masks=[m.numpy() for m in self.masks],
            class_labels=self.class_labels,
            scores=self.scores.cpu().numpy() if isinstance(self.scores, torch.Tensor) else self.scores,
            semantics=self.semantics.cpu().numpy() if isinstance(self.semantics, torch.Tensor) else self.semantics,
            panoptic=self.panoptic.cpu().numpy() if isinstance(self.panoptic, torch.Tensor) else self.panoptic
        )
    
    def visualize(
        self,
        color_map: Optional[Dict[str, Tuple[int, int, int]]] = None,
        alpha: float = 0.5
    ) -> np.ndarray:
        """Visualize segmentation masks overlayed on image.
        
        Args:
            color_map: Dictionary mapping labels to RGB colors
            alpha: Transparency of the overlay
            
        Returns:
            RGB visualization image
        """
        import cv2
        
        overlay = np.zeros((*self.image_size, 3), dtype=np.uint8)
        
        if color_map is None:
            import matplotlib.cm as cm
            colors = cm.tab20(np.linspace(0, 1, max(20, len(self.masks))))
            color_map = {}
            labels = [m.label for m in self.masks if m.label]
            unique_labels = list(dict.fromkeys(labels))
            for i, label in enumerate(unique_labels):
                color_map[label] = tuple(int(c * 255) for c in colors[i % 20][:3])
        
        for i, mask in enumerate(self.masks):
            if isinstance(mask.segmentation, torch.Tensor):
                seg = mask.segmentation.cpu().numpy()
            else:
                seg = mask.segmentation
            
            if seg.ndim == 3:
                seg = seg.any(axis=-1)
            
            color = color_map.get(mask.label, (128, 128, 128))
            
            mask_bool = seg.astype(bool)
            overlay[mask_bool] = color
        
        return overlay