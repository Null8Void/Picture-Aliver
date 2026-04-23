"""Image2Video AI - Production Image to Video Synthesis System"""

__version__ = "1.0.0"
__author__ = "AI Engineering Team"

from .pipeline import Image2VideoPipeline
from .config import Config
from .device import DeviceManager

__all__ = [
    "Image2VideoPipeline",
    "Config",
    "DeviceManager",
]