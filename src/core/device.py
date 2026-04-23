"""Device management for GPU/CPU acceleration."""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class DeviceInfo:
    """Information about available compute device."""
    name: str
    type: str  # cuda, mps, cpu
    memory_total: Optional[int] = None
    memory_available: Optional[int] = None
    compute_capability: Optional[Tuple[int, int]] = None
    driver_version: Optional[str] = None
    
    @property
    def is_cuda(self) -> bool:
        return self.type == "cuda"
    
    @property
    def is_mps(self) -> bool:
        return self.type == "mps"
    
    @property
    def is_cpu(self) -> bool:
        return self.type == "cpu"
    
    @property
    def memory_gb(self) -> Optional[float]:
        if self.memory_total:
            return self.memory_total / (1024**3)
        return None


class DeviceManager:
    """Manages compute devices and device selection for the pipeline."""
    
    _instance: Optional["DeviceManager"] = None
    
    def __new__(cls) -> "DeviceManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._devices: dict[str, DeviceInfo] = {}
        self._current_device: Optional[DeviceInfo] = None
        self._device_name = "cpu"
        self._initialized = True
        self._detect_devices()
    
    def _detect_devices(self) -> None:
        """Detect all available compute devices."""
        self._devices["cpu"] = DeviceInfo(
            name="CPU",
            type="cpu"
        )
        
        if torch.cuda.is_available():
            try:
                device_props = torch.cuda.get_device_properties(0)
                self._devices["cuda"] = DeviceInfo(
                    name=device_props.name,
                    type="cuda",
                    memory_total=device_props.total_memory,
                    compute_capability=(
                        device_props.major,
                        device_props.minor
                    ),
                    driver_version=torch.version.cuda
                )
            except Exception:
                pass
        
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._devices["mps"] = DeviceInfo(
                name="Apple Silicon GPU",
                type="mps"
            )
        
        self._current_device = self._devices.get("cuda", self._devices.get("cpu"))
    
    def get_device(self, device: Optional[str] = None) -> torch.device:
        """Get PyTorch device object.
        
        Args:
            device: Device name ('cuda', 'mps', 'cpu') or None for auto
            
        Returns:
            torch.device object
        """
        if device is None:
            device = self._resolve_auto_device()
        
        return torch.device(device)
    
    def _resolve_auto_device(self) -> str:
        """Resolve the best available device automatically."""
        if "cuda" in self._devices:
            return "cuda"
        elif "mps" in self._devices:
            return "mps"
        return "cpu"
    
    def set_device(self, device: str) -> None:
        """Set the current compute device.
        
        Args:
            device: Device name ('cuda', 'mps', 'cpu')
        """
        if device not in self._devices:
            raise ValueError(
                f"Device '{device}' not available. "
                f"Available devices: {list(self._devices.keys())}"
            )
        
        self._current_device = self._devices[device]
        self._device_name = device
    
    @property
    def current_device(self) -> DeviceInfo:
        """Get current device information."""
        if self._current_device is None:
            self._current_device = self._devices.get(
                self._resolve_auto_device(),
                DeviceInfo(name="CPU", type="cpu")
            )
        return self._current_device
    
    @property
    def available_devices(self) -> dict[str, DeviceInfo]:
        """Get all available devices."""
        return self._devices.copy()
    
    @property
    def device_name(self) -> str:
        """Get current device name string."""
        return self._device_name
    
    def memory_stats(self) -> dict:
        """Get memory statistics for current device."""
        if self._device_name == "cuda" and torch.cuda.is_available():
            return {
                "total": torch.cuda.get_device_properties(0).total_memory,
                "allocated": torch.cuda.memory_allocated(),
                "reserved": torch.cuda.memory_reserved(),
                "free": torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(),
            }
        return {}
    
    def clear_cache(self) -> None:
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def synchronize(self) -> None:
        """Synchronize all devices."""
        if self._device_name == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
    
    def enable_tf32(self) -> None:
        """Enable TF32 precision on Ampere+ GPUs for faster computation."""
        if self._device_name == "cuda":
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass
    
    def disable_tf32(self) -> None:
        """Disable TF32 precision."""
        if self._device_name == "cuda":
            try:
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
            except Exception:
                pass
    
    def __repr__(self) -> str:
        return (
            f"DeviceManager(available={list(self._devices.keys())}, "
            f"current={self._device_name})"
        )


_device_manager = DeviceManager()


def get_device_manager() -> DeviceManager:
    """Get the global device manager instance."""
    return _device_manager


def get_torch_device(device: Optional[str] = None) -> torch.device:
    """Convenience function to get a torch device."""
    return _device_manager.get_device(device)


def get_optimal_device() -> str:
    """Get the optimal device string."""
    return _device_manager._resolve_auto_device()