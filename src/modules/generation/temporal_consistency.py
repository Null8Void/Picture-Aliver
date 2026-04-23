"""Temporal consistency management for smooth video generation."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConsistencyManager:
    """Manages temporal consistency for video generation.
    
    Provides:
    - Temporal smoothing of generated frames
    - Cross-frame attention mechanisms
    - Loop detection and optimization
    - Motion coherence enforcement
    """
    
    def __init__(
        self,
        num_frames: int = 24,
        temporal_window: int = 3,
        spatial_blur_sigma: float = 0.5,
        temporal_blur_sigma: float = 1.0,
        device: Optional[torch.device] = None
    ):
        self.num_frames = num_frames
        self.temporal_window = temporal_window
        self.spatial_blur_sigma = spatial_blur_sigma
        self.temporal_blur_sigma = temporal_blur_sigma
        self.device = device or torch.device("cpu")
        
        self.history: List[torch.Tensor] = []
        self.motion_field: Optional[torch.Tensor] = None
        self.attention_cache: Optional[torch.Tensor] = None
    
    def temporal_smooth(
        self,
        frames: torch.Tensor,
        method: str = "gaussian"
    ) -> torch.Tensor:
        """Apply temporal smoothing to video frames.
        
        Args:
            frames: Video tensor [T, C, H, W]
            method: Smoothing method ('gaussian', 'bilateral', 'median')
            
        Returns:
            Smoothed frames [T, C, H, W]
        """
        if method == "gaussian":
            return self._gaussian_smooth(frames)
        elif method == "bilateral":
            return self._bilateral_smooth(frames)
        elif method == "median":
            return self._median_smooth(frames)
        else:
            return frames
    
    def _gaussian_smooth(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian temporal smoothing."""
        if self.temporal_blur_sigma == 0:
            return frames
        
        T, C, H, W = frames.shape
        
        kernel_size = int(2 * np.ceil(2 * self.temporal_blur_sigma) + 1)
        kernel_size = max(3, kernel_size)
        
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        half = kernel_size // 2
        
        padded = torch.cat([
            frames[-half:],
            frames,
            frames[:half]
        ], dim=0)
        
        temporal_kernel = self._get_gaussian_kernel(
            kernel_size,
            self.temporal_blur_sigma
        ).view(1, 1, kernel_size, 1, 1).to(frames.device)
        
        if self.spatial_blur_sigma > 0:
            spatial_kernel = self._get_gaussian_kernel(
                kernel_size,
                self.spatial_blur_sigma
            ).view(1, 1, 1, kernel_size, kernel_size).to(frames.device)
            
            smoothed = F.conv3d(
                padded.unsqueeze(0),
                spatial_kernel.expand(C, C, -1, -1, -1),
                padding=kernel_size // 2,
                groups=C
            )
            
            smoothed = F.conv3d(
                smoothed,
                temporal_kernel.expand(1, 1, -1, 1, 1),
                padding=kernel_size // 2,
                groups=1
            ).squeeze(0)
        else:
            smoothed = F.conv3d(
                padded.unsqueeze(0),
                temporal_kernel,
                padding=kernel_size // 2
            ).squeeze(0)
        
        return smoothed
    
    def _get_gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """Create Gaussian kernel."""
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        kernel = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        return kernel
    
    def _bilateral_smooth(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply bilateral temporal filtering."""
        T, C, H, W = frames.shape
        smoothed = frames.clone()
        
        for t in range(T):
            neighbors = []
            weights = []
            
            for dt in range(-self.temporal_window, self.temporal_window + 1):
                if dt == 0:
                    continue
                
                nt = t + dt
                if 0 <= nt < T:
                    neighbors.append(frames[nt])
                    temporal_dist = abs(dt) / self.temporal_window
                    intensity_dist = torch.mean(
                        torch.abs(frames[t] - frames[nt]),
                        dim=[1, 2, 3],
                        keepdim=True
                    )
                    weight = torch.exp(
                        -temporal_dist / 2 - intensity_dist.pow(2) / (2 * 0.1)
                    )
                    weights.append(weight)
            
            if neighbors:
                neighbors_tensor = torch.stack(neighbors, dim=0)
                weights_tensor = torch.stack(weights, dim=0)
                
                weights_tensor = weights_tensor / weights_tensor.sum(dim=0, keepdim=True)
                
                weighted_sum = (neighbors_tensor * weights_tensor).sum(dim=0)
                smoothed[t] = 0.8 * frames[t] + 0.2 * weighted_sum
        
        return smoothed
    
    def _median_smooth(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply temporal median filtering."""
        T, C, H, W = frames.shape
        smoothed = frames.clone()
        
        for t in range(T):
            window = []
            
            for dt in range(-self.temporal_window, self.temporal_window + 1):
                nt = t + dt
                if 0 <= nt < T:
                    window.append(frames[nt])
            
            window_tensor = torch.stack(window, dim=0)
            
            median = torch.median(window_tensor, dim=0)[0]
            smoothed[t] = 0.7 * frames[t] + 0.3 * median
        
        return smoothed
    
    def enforce_loop_consistency(
        self,
        frames: torch.Tensor,
        loop_frames: int = 2,
        strength: float = 0.5
    ) -> torch.Tensor:
        """Enforce loop consistency between first and last frames.
        
        Args:
            frames: Video tensor [T, C, H, W]
            loop_frames: Number of frames to blend
            strength: Blend strength
            
        Returns:
            Loop-consistent frames
        """
        T = frames.shape[0]
        result = frames.clone()
        
        for i in range(loop_frames):
            blend = strength * (i + 1) / loop_frames
            result[T - 1 - i] = (
                (1 - blend) * frames[T - 1 - i] + 
                blend * frames[i]
            )
            result[i] = (
                (1 - blend) * frames[i] + 
                blend * frames[T - 1 - i]
            )
        
        return result
    
    def compute_temporal_attention(
        self,
        frames: torch.Tensor,
        num_heads: int = 8
    ) -> torch.Tensor:
        """Compute cross-frame attention for consistency.
        
        Args:
            frames: Video tensor [T, C, H, W]
            num_heads: Number of attention heads
            
        Returns:
            Attention weights [T, T]
        """
        T, C, H, W = frames.shape
        
        features = frames.reshape(T, C, H * W).transpose(1, 2)
        
        batch_size = 1
        seq_len = H * W
        
        d_k = C // num_heads
        
        q = self._linear(features, d_k * num_heads)
        k = self._linear(features, d_k * num_heads)
        v = self._linear(features, d_k * num_heads)
        
        q = q.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
        attention = F.softmax(scores, dim=-1)
        
        return attention.mean(dim=[0, 1])
    
    def _linear(self, x: torch.Tensor, out_features: int) -> torch.Tensor:
        """Simple linear projection."""
        in_features = x.shape[-1]
        weight = torch.randn(in_features, out_features, device=x.device)
        bias = torch.zeros(out_features, device=x.device)
        return F.linear(x, weight, bias)
    
    def propagate_motion(
        self,
        initial_frame: torch.Tensor,
        flow: torch.Tensor,
        num_frames: int
    ) -> List[torch.Tensor]:
        """Propagate motion to generate subsequent frames.
        
        Args:
            initial_frame: Starting frame [C, H, W]
            flow: Motion flow field [T, 2, H, W]
            num_frames: Number of frames to generate
            
        Returns:
            List of generated frames
        """
        frames = [initial_frame]
        current = initial_frame
        
        for t in range(num_frames - 1):
            flow_t = flow[t] if t < len(flow) else flow[-1]
            
            warped = self._warp_frame(current, flow_t)
            frames.append(warped)
            current = warped
        
        return frames
    
    def _warp_frame(
        self,
        frame: torch.Tensor,
        flow: torch.Tensor
    ) -> torch.Tensor:
        """Warp frame using flow field."""
        B, C, H, W = frame.shape if frame.dim() == 4 else (1, *frame.shape)
        
        if frame.dim() == 3:
            frame = frame.unsqueeze(0)
        
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=frame.device, dtype=torch.float32),
            torch.arange(W, device=frame.device, dtype=torch.float32),
            indexing='ij'
        )
        
        grid_x = grid_x + flow[0].unsqueeze(0)
        grid_y = grid_y + flow[1].unsqueeze(0)
        
        grid_x = 2.0 * grid_x / (W - 1) - 1.0
        grid_y = 2.0 * grid_y / (H - 1) - 1.0
        
        grid = torch.stack([grid_x, grid_y], dim=-1)
        grid = grid.unsqueeze(0)
        
        warped = F.grid_sample(
            frame,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        
        return warped.squeeze(0)
    
    def reduce_flickering(
        self,
        frames: torch.Tensor,
        threshold: float = 0.1,
        correction_strength: float = 0.3
    ) -> torch.Tensor:
        """Detect and reduce flickering artifacts.
        
        Args:
            frames: Video tensor [T, C, H, W]
            threshold: Difference threshold for flicker detection
            correction_strength: Strength of flicker correction
            
        Returns:
            Flicker-reduced frames
        """
        T, C, H, W = frames.shape
        result = frames.clone()
        
        for t in range(1, T - 1):
            diff_prev = torch.abs(frames[t] - frames[t - 1]).mean()
            diff_next = torch.abs(frames[t + 1] - frames[t]).mean()
            
            avg_diff = (diff_prev + diff_next) / 2
            
            if diff_prev > threshold and diff_next > threshold:
                correction = frames[t - 1] + (frames[t + 1] - frames[t - 1]) / 2
                result[t] = (
                    (1 - correction_strength) * frames[t] +
                    correction_strength * correction
                )
        
        return result
    
    def stabilize_frames(
        self,
        frames: torch.Tensor,
        smooth_path: bool = True,
        window_size: int = 5
    ) -> torch.Tensor:
        """Stabilize frames by reducing camera shake.
        
        Args:
            frames: Video tensor [T, C, H, W]
            smooth_path: Whether to smooth trajectory
            window_size: Smoothing window size
            
        Returns:
            Stabilized frames
        """
        T, C, H, W = frames.shape
        
        transforms = []
        prev_frame = frames[0]
        
        for t in range(T):
            current_frame = frames[t]
            
            diff = self._estimate_translation(prev_frame, current_frame)
            transforms.append(diff)
            
            prev_frame = current_frame
        
        transforms = torch.tensor(transforms, device=frames.device)
        
        if smooth_path:
            transforms = transforms.float()
            kernel = torch.ones(window_size, device=frames.device) / window_size
            transforms[:, 0] = F.conv1d(
                transforms[:, 0].unsqueeze(0).unsqueeze(0),
                kernel.unsqueeze(0).unsqueeze(0),
                padding=window_size // 2
            ).squeeze()
            transforms[:, 1] = F.conv1d(
                transforms[:, 1].unsqueeze(0).unsqueeze(0),
                kernel.unsqueeze(0).unsqueeze(0),
                padding=window_size // 2
            ).squeeze()
        
        stabilized = []
        cumulative = torch.zeros(2, device=frames.device)
        
        for t in range(T):
            cumulative += transforms[t]
            
            grid = self._create_shift_grid(
                H, W,
                -cumulative[0],
                -cumulative[1],
                frames.device
            )
            
            stabilized.append(
                F.grid_sample(
                    frames[t].unsqueeze(0),
                    grid,
                    align_corners=True
                ).squeeze(0)
            )
        
        return torch.stack(stabilized, dim=0)
    
    def _estimate_translation(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor
    ) -> Tuple[float, float]:
        """Estimate translation between frames."""
        diff = (frame1 - frame2).mean(dim=0)
        
        grad_x = diff[:, 1:] - diff[:, :-1]
        grad_y = diff[1:, :] - diff[:-1, :]
        
        dx = grad_x.mean().item() * 2
        dy = grad_y.mean().item() * 2
        
        return (dx, dy)
    
    def _create_shift_grid(
        self,
        H: int,
        W: int,
        dx: float,
        dy: float,
        device: torch.device
    ) -> torch.Tensor:
        """Create grid for frame shifting."""
        grid_x = torch.linspace(-1, 1, W, device=device) + dx / W * 2
        grid_y = torch.linspace(-1, 1, H, device=device) + dy / H * 2
        
        grid_x, grid_y = torch.meshgrid(grid_x, grid_y, indexing='x')
        
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        
        return grid


class MotionPropagator:
    """Propagates motion from sparse keyframes to dense frames."""
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cpu")
        self.flow_estimators: List[nn.Module] = []
    
    def propagate_from_keyframes(
        self,
        keyframes: torch.Tensor,
        keyframe_indices: torch.Tensor,
        num_output_frames: int
    ) -> torch.Tensor:
        """Propagate motion from sparse keyframes.
        
        Args:
            keyframes: Keyframe images [K, C, H, W]
            keyframe_indices: Indices of keyframes [K]
            num_output_frames: Total output frames
            
        Returns:
            Dense frames [T, C, H, W]
        """
        K = len(keyframes)
        T = num_output_frames
        
        frames = torch.zeros(T, *keyframes.shape[1:], device=self.device)
        
        frames[keyframe_indices] = keyframes
        
        for k in range(K - 1):
            start_idx = keyframe_indices[k].item()
            end_idx = keyframe_indices[k + 1].item()
            
            start_frame = keyframes[k]
            end_frame = keyframes[k + 1]
            
            segment_frames = self._interpolate_segment(
                start_frame,
                end_frame,
                end_idx - start_idx + 1
            )
            
            for i, frame_idx in enumerate(range(start_idx, end_idx + 1)):
                if frame_idx < T:
                    frames[frame_idx] = segment_frames[i]
        
        return frames
    
    def _interpolate_segment(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        num_frames: int
    ) -> List[torch.Tensor]:
        """Interpolate between two frames."""
        frames = []
        for i in range(num_frames):
            alpha = i / (num_frames - 1) if num_frames > 1 else 0
            
            eased = self._ease_function(alpha)
            
            frame = (1 - eased) * start + eased * end
            
            frames.append(frame)
        
        return frames
    
    def _ease_function(self, t: float) -> float:
        """Apply easing function for natural motion."""
        return t * t * (3 - 2 * t)


class NoiseScheduler:
    """Manages noise scheduling for temporal consistency."""
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear"
    ):
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        
        self.betas = self._get_betas()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
    
    def _get_betas(self) -> torch.Tensor:
        """Get beta schedule."""
        if self.beta_schedule == "linear":
            return torch.linspace(self.beta_start, self.beta_end, self.num_train_timesteps)
        elif self.beta_schedule == "quadratic":
            return torch.linspace(self.beta_start ** 0.5, self.beta_end ** 0.5, self.num_train_timesteps) ** 2
        elif self.beta_schedule == "scaled_linear":
            return torch.linspace(self.beta_start ** 0.5, self.beta_end ** 0.5, self.num_train_timesteps) ** 2
        else:
            return torch.linspace(self.beta_start, self.beta_end, self.num_train_timesteps)
    
    def add_noise(
        self,
        original: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Add noise to original samples."""
        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten().reshape(-1, 1, 1, 1)
        
        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten().reshape(-1, 1, 1, 1)
        
        noisy = sqrt_alpha_prod * original + sqrt_one_minus_alpha_prod * noise
        
        return noisy
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor
    ) -> torch.Tensor:
        """Perform one denoising step."""
        t = timestep
        
        pred_original_sample = (
            sample - (1 - self.alphas[t]) ** 0.5 * model_output
        ) / self.alphas[t] ** 0.5
        
        pred_original_sample = pred_original_sample.clamp(-1, 1)
        
        prev_t = t - 1 if t > 0 else 0
        
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0)
        
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        predicted_prev_sample = (
            alpha_prod_t_prev ** 0.5 * self.alphas[t] ** 0.5 * pred_original_sample +
            (1 - alpha_prod_t_prev) ** 0.5 * (1 - self.alphas[t]) ** 0.5 * model_output
        )
        
        variance = (beta_prod_t / beta_prod_t_prev) * (1 - alpha_prod_t_prev) / (1 - alpha_prod_t)
        
        return predicted_prev_sample