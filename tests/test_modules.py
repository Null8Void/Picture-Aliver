"""Tests for pipeline modules."""

import pytest
import torch
import numpy as np
from PIL import Image


class TestPipelineConfig:
    """Tests for pipeline configuration."""
    
    def test_default_config(self):
        from src.core.pipeline import PipelineConfig
        
        config = PipelineConfig()
        assert config.enable_depth is True
        assert config.enable_segmentation is True
        assert config.enable_motion is True
        assert config.enable_consistency is True
    
    def test_custom_config(self):
        from src.core.pipeline import PipelineConfig
        
        config = PipelineConfig(
            enable_depth=False,
            motion_mode="camera",
            output_format="webm"
        )
        assert config.enable_depth is False
        assert config.motion_mode == "camera"


class TestVideoFrames:
    """Tests for video frames container."""
    
    def test_create_frames(self):
        from src.modules.generation import VideoFrames
        
        frames = VideoFrames()
        assert len(frames) == 0
        assert frames.num_frames == 0
    
    def test_append_frame(self):
        from src.modules.generation import VideoFrames
        
        frames = VideoFrames()
        tensor = torch.rand(3, 64, 64)
        frames.append(tensor)
        
        assert len(frames) == 1
        assert frames.num_frames == 1
    
    def test_shape(self):
        from src.modules.generation import VideoFrames
        
        frames = VideoFrames()
        tensor = torch.rand(3, 64, 64)
        frames.append(tensor)
        
        assert frames.shape == (3, 64, 64)


class TestDepthMap:
    """Tests for depth map handling."""
    
    def test_create_depth_map(self):
        from src.modules.depth import DepthMap
        
        depth = torch.rand(64, 64)
        dm = DepthMap(depth=depth)
        
        assert dm.depth.shape == (64, 64)
        assert dm.normalized is not None


class TestSegmentationMask:
    """Tests for segmentation masks."""
    
    def test_create_mask(self):
        from src.modules.segmentation import Mask
        
        seg = np.zeros((64, 64), dtype=bool)
        seg[20:40, 20:40] = True
        
        mask = Mask(segmentation=seg)
        assert mask.area > 0
        assert mask.bbox is not None


class TestFlowField:
    """Tests for optical flow field."""
    
    def test_create_flow(self):
        from src.modules.motion import FlowField
        
        flow = np.zeros((64, 64, 2), dtype=np.float32)
        flow[..., 0] = 1.0
        flow[..., 1] = 1.0
        
        ff = FlowField(flow=flow)
        assert ff.magnitude is not None
        assert ff.angle is not None
    
    def test_magnitude(self):
        from src.modules.motion import FlowField
        
        flow = np.array([[[1.0, 0.0]]])
        ff = FlowField(flow=flow)
        
        assert np.isclose(ff.magnitude.item(), 1.0, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])