"""Image utilities for preprocessing and postprocessing."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image


def load_image(
    path: Union[str, Image.Image, np.ndarray, torch.Tensor],
    size: Optional[Tuple[int, int]] = None,
    mode: str = "RGB"
) -> Image.Image:
    """Load image from various sources.
    
    Args:
        path: Image path, PIL Image, numpy array, or tensor
        size: Optional target size (H, W)
        mode: PIL color mode
        
    Returns:
        PIL Image
    """
    if isinstance(path, Image.Image):
        image = path
    elif isinstance(path, str):
        image = Image.open(path)
    elif isinstance(path, np.ndarray):
        if path.dtype == np.float32 or path.dtype == np.float64:
            if path.max() <= 1:
                image = Image.fromarray((path * 255).astype(np.uint8))
            else:
                image = Image.fromarray(path.astype(np.uint8))
        else:
            image = Image.fromarray(path)
    elif isinstance(path, torch.Tensor):
        image = tensor_to_pil(path)
    else:
        raise ValueError(f"Unsupported image type: {type(path)}")
    
    if mode and image.mode != mode:
        image = image.convert(mode)
    
    if size:
        image = resize_image(image, size)
    
    return image


def save_image(
    image: Union[Image.Image, np.ndarray, torch.Tensor],
    path: str,
    quality: int = 95
) -> None:
    """Save image to file.
    
    Args:
        image: Image to save
        path: Output path
        quality: JPEG quality
    """
    if isinstance(image, torch.Tensor):
        image = tensor_to_pil(image)
    elif isinstance(image, np.ndarray):
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = Image.fromarray((image * 255).astype(np.uint8))
        elif image.max() > 255:
            image = Image.fromarray(image.astype(np.uint8))
        else:
            image = Image.fromarray(image)
    
    if not isinstance(image, Image.Image):
        raise ValueError(f"Cannot convert {type(image)} to PIL Image")
    
    path = str(path)
    if path.endswith('.jpg') or path.endswith('.jpeg'):
        image.save(path, quality=quality, optimize=True)
    else:
        image.save(path)


def resize_image(
    image: Union[Image.Image, np.ndarray],
    size: Tuple[int, int],
    mode: str = "bilinear",
    aspect_ratio: bool = True
) -> Image.Image:
    """Resize image to target size.
    
    Args:
        image: Input image
        size: Target size (H, W)
        mode: Interpolation mode
        aspect_ratio: Preserve aspect ratio by using largest dimension
        
    Returns:
        Resized image
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    if not isinstance(image, Image.Image):
        raise Image
    
    target_h, target_w = size
    
    if aspect_ratio:
        image_w, image_h = image.size
        scale = min(target_w / image_w, target_h / image_h)
        new_w = int(image_w * scale)
        new_h = int(image_h * scale)
        size = (new_h, new_w)
    
    resample = {
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
        "nearest": Image.Resampling.NEAREST,
    }.get(mode, Image.Resampling.BILINEAR)
    
    return image.resize((size[1], size[0]), resample)


def center_crop(
    image: Union[Image.Image, np.ndarray],
    size: Tuple[int, int]
) -> Union[Image.Image, np.ndarray]:
    """Center crop image to target size.
    
    Args:
        image: Input image
        size: Crop size (H, W)
        
    Returns:
        Cropped image
    """
    target_h, target_w = size
    
    if isinstance(image, np.ndarray):
        h, w = image.shape[:2]
        start_y = (h - target_h) // 2
        start_x = (w - target_w) // 2
        
        cropped = image[start_y:start_y + target_h, start_x:start_x + target_w]
        return cropped
    else:
        w, h = image.size
        start_y = (h - target_h) // 2
        start_x = (w - target_w) // 2
        
        return image.crop((start_x, start_y, start_x + target_w, start_y + target_h))


def normalize_image(
    image: Union[np.ndarray, torch.Tensor],
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)
) -> Union[np.ndarray, torch.Tensor]:
    """Normalize image with mean and std.
    
    Args:
        image: Input image
        mean: Mean values per channel
        std: Std values per channel
        
    Returns:
        Normalized image
    """
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
    
    if image.dtype == torch.uint8:
        image = image.float() / 255.0
    
    if image.dim() == 4:
        image = image.squeeze(0)
    
    if image.shape[0] in [1, 3]:
        image = image.transpose(0, 2).transpose(1, 0)
    
    mean_t = torch.tensor(mean, dtype=image.dtype, device=image.device).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=image.dtype, device=image.device).view(-1, 1, 1)
    
    return (image - mean_t) / std_t


def denormalize_image(
    image: Union[np.ndarray, torch.Tensor],
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)
) -> Union[np.ndarray, torch.Tensor]:
    """Denormalize image.
    
    Args:
        image: Normalized image
        mean: Mean used for normalization
        std: Std used for normalization
        
    Returns:
        Denormalized image
    """
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
    
    mean_t = torch.tensor(mean, dtype=image.dtype, device=image.device).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=image.dtype, device=image.device).view(-1, 1, 1)
    
    result = image * std_t + mean_t
    
    if result.dim() == 3:
        result = result.transpose(0, 2).transpose(0, 1)
    
    return result.clamp(0, 1)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image.
    
    Args:
        tensor: Input tensor [C, H, W] or [B, C, H, W]
        
    Returns:
        PIL Image
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    if tensor.shape[0] == 3:
        tensor = tensor.transpose(0, 2).transpose(0, 1)
    
    array = tensor.cpu().numpy()
    
    if array.dtype == np.float32 or array.dtype == np.float64:
        array = np.clip(array, 0, 1)
        array = (array * 255).astype(np.uint8)
    elif array.max() > 255:
        array = array.astype(np.uint8)
    
    return Image.fromarray(array)


def pil_to_tensor(image: Image.Image, normalize: bool = False) -> torch.Tensor:
    """Convert PIL Image to tensor.
    
    Args:
        image: PIL Image
        normalize: Whether to normalize to [0, 1]
        
    Returns:
        Tensor [C, H, W]
    """
    array = np.array(image)
    
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    
    if array.dtype == np.uint8:
        tensor = torch.from_numpy(array).permute(2, 0, 1).float()
        if normalize:
            tensor = tensor / 255.0
    else:
        tensor = torch.from_numpy(array).permute(2, 0, 1).float()
    
    return tensor


def create_image_grid(
    images: list,
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    padding: int = 2
) -> Image.Image:
    """Create a grid of images.
    
    Args:
        images: List of images
        rows: Number of rows
        cols: Number of columns
        padding: Padding between images
        
    Returns:
        Grid image
    """
    num_images = len(images)
    
    if rows is None and cols is None:
        cols = int(np.ceil(np.sqrt(num_images)))
        rows = int(np.ceil(num_images / cols))
    elif rows is None:
        rows = int(np.ceil(num_images / cols))
    elif cols is None:
        cols = int(np.ceil(num_images / rows))
    
    for i, img in enumerate(images):
        if isinstance(img, torch.Tensor):
            images[i] = tensor_to_pil(img)
        elif isinstance(img, np.ndarray):
            if img.dtype == np.float32 or img.dtype == np.float64:
                images[i] = Image.fromarray((img * 255).astype(np.uint8))
            else:
                images[i] = Image.fromarray(img)
    
    img_w, img_h = images[0].size
    grid_w = cols * img_w + (cols + 1) * padding
    grid_h = rows * img_h + (rows + 1) * padding
    
    grid = Image.new('RGB', (grid_w, grid_h), color=(128, 128, 128))
    
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        
        x = col * img_w + (col + 1) * padding
        y = row * img_h + (row + 1) * padding
        
        grid.paste(img, (x, y))
    
    return grid