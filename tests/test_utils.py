"""Tests for utility functions."""

import pytest
import torch
import numpy as np
from PIL import Image

from src.utils.image_utils import (
    load_image,
    save_image,
    resize_image,
    center_crop,
    normalize_image,
    denormalize_image,
    tensor_to_pil,
    pil_to_tensor
)


class TestImageUtils:
    """Tests for image utilities."""
    
    def test_load_image_from_array(self):
        arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img = load_image(arr)
        assert isinstance(img, Image.Image)
    
    def test_resize_image(self):
        img = Image.new('RGB', (100, 100))
        resized = resize_image(img, (50, 50))
        assert resized.size == (50, 50)
    
    def test_center_crop(self):
        img = Image.new('RGB', (100, 100))
        cropped = center_crop(img, (50, 50))
        assert cropped.size == (50, 50)
    
    def test_normalize_denormalize(self):
        tensor = torch.rand(3, 100, 100)
        normalized = normalize_image(tensor)
        denormalized = denormalize_image(normalized)
        
        assert denormalized.shape == tensor.shape


class TestVideoUtils:
    """Tests for video utilities."""
    
    def test_frames_to_list(self):
        from src.modules.generation import VideoFrames
        
        frames = VideoFrames()
        for _ in range(5):
            frames.append(torch.rand(3, 64, 64))
        
        frame_list = frames.to_list()
        assert len(frame_list) == 5
        assert all(isinstance(f, np.ndarray) for f in frame_list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])