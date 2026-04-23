"""Video utilities for frame manipulation and video processing."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image


def frames_to_video(
    frames: Union[List[np.ndarray], List[Image.Image], torch.Tensor],
    output_path: Union[str, Path],
    fps: int = 24,
    codec: str = "libx264",
    quality: int = 23,
    bitrate: str = "5M"
) -> None:
    """Convert frames to video file.
    
    Args:
        frames: List of frames or video tensor [T, C, H, W]
        output_path: Output video path
        fps: Frames per second
        codec: Video codec
        quality: Quality (lower is better, 18-28 typical)
        bitrate: Video bitrate
    """
    try:
        import cv2
        _frames_to_video_cv2(frames, output_path, fps, quality)
    except ImportError:
        try:
            _frames_to_video_ffmpeg(frames, output_path, fps, codec, bitrate)
        except ImportError:
            _frames_to_video_pil(frames, output_path)


def _frames_to_video_cv2(
    frames: Union[List[np.ndarray], torch.Tensor],
    output_path: Union[str, Path],
    fps: int,
    quality: int
) -> None:
    """Convert frames to video using OpenCV."""
    import cv2
    
    if isinstance(frames, torch.Tensor):
        frames = tensor_to_frames_list(frames)
    
    if not frames:
        return
    
    first_frame = frames[0]
    if isinstance(first_frame, Image.Image):
        first_frame = np.array(first_frame)
    
    h, w = first_frame.shape[:2]
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (w, h)
    )
    
    for frame in frames:
        if isinstance(frame, Image.Image):
            frame = np.array(frame)
        
        if frame.shape[-1] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        writer.write(frame)
    
    writer.release()


def _frames_to_video_ffmpeg(
    frames: Union[List[np.ndarray], torch.Tensor],
    output_path: Union[str, Path],
    fps: int,
    codec: str,
    bitrate: str
) -> None:
    """Convert frames to video using FFmpeg."""
    import subprocess
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    temp_dir = output_path.parent / f".temp_frames_{output_path.stem}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    for i, frame in enumerate(frames):
        if isinstance(frame, torch.Tensor):
            frame = tensor_to_np(frame)
        if isinstance(frame, Image.Image):
            frame = np.array(frame)
        
        frame_path = temp_dir / f"frame_{i:06d}.png"
        Image.fromarray(frame).save(frame_path)
    
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(temp_dir / "frame_%06d.png"),
        "-c:v", codec,
        "-b:v", bitrate,
        "-pix_fmt", "yuv420p",
        str(output_path)
    ]
    
    subprocess.run(cmd, check=True, capture_output=True)
    
    for f in temp_dir.glob("*.png"):
        f.unlink()
    temp_dir.rmdir()


def _frames_to_video_pil(
    frames: List[Image.Image],
    output_path: Union[str, Path]
) -> None:
    """Save frames as individual images when no video encoder available."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    for i, frame in enumerate(frames):
        if isinstance(frame, torch.Tensor):
            frame = tensor_to_pil(frame)
        frame.save(output_path.parent / f"{output_path.stem}_{i:04d}.png")


def tensor_to_frames_list(tensor: torch.Tensor) -> List[np.ndarray]:
    """Convert video tensor to list of numpy arrays."""
    if tensor.dim() == 4:
        T, C, H, W = tensor.shape
        frames = []
        for i in range(T):
            frame = tensor[i]
            if C == 3:
                frame = frame.permute(1, 2, 0)
            frames.append(frame.cpu().numpy())
        return frames
    return [tensor.cpu().numpy()]


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    if tensor.shape[0] == 3:
        tensor = tensor.transpose(0, 2).transpose(0, 1)
    
    array = tensor.cpu().numpy()
    array = np.clip(array, 0, 1)
    array = (array * 255).astype(np.uint8)
    
    return Image.fromarray(array)


def tensor_to_np(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy array."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    if tensor.shape[0] == 3:
        tensor = tensor.transpose(0, 2).transpose(0, 1)
    
    array = tensor.cpu().numpy()
    
    if array.dtype == np.float32 or array.dtype == np.float64:
        array = np.clip(array, 0, 1)
        array = (array * 255).astype(np.uint8)
    
    return array


def video_to_frames(
    video_path: Union[str, Path],
    max_frames: Optional[int] = None,
    step: int = 1
) -> List[np.ndarray]:
    """Extract frames from video file.
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to extract
        step: Extract every nth frame
        
    Returns:
        List of frame arrays
    """
    try:
        import cv2
        return _video_to_frames_cv2(video_path, max_frames, step)
    except ImportError:
        raise ImportError("OpenCV or ffmpeg is required for video reading")


def _video_to_frames_cv2(
    video_path: Union[str, Path],
    max_frames: Optional[int],
    step: int
) -> List[np.ndarray]:
    """Extract frames using OpenCV."""
    import cv2
    
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    
    frames = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        if frame_idx % step == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            
            if max_frames and len(frames) >= max_frames:
                break
        
        frame_idx += 1
    
    cap.release()
    
    return frames


def resize_frames(
    frames: List[np.ndarray],
    size: Tuple[int, int],
    mode: str = "bilinear"
) -> List[np.ndarray]:
    """Resize list of frames.
    
    Args:
        frames: List of frames
        size: Target size (H, W)
        mode: Interpolation mode
        
    Returns:
        Resized frames
    """
    import cv2
    
    target_h, target_w = size
    resized = []
    
    for frame in frames:
        resized_frame = cv2.resize(
            frame,
            (target_w, target_h),
            interpolation=getattr(cv2, f'INTER_{mode.upper()}', cv2.INTER_LINEAR)
        )
        resized.append(resized_frame)
    
    return resized


def blend_frames(
    frame1: np.ndarray,
    frame2: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """Blend two frames.
    
    Args:
        frame1: First frame
        frame2: Second frame
        alpha: Blend factor (0=frame1, 1=frame2)
        
    Returns:
        Blended frame
    """
    if frame1.dtype != np.float32:
        frame1 = frame1.astype(np.float32) / 255.0
    if frame2.dtype != np.float32:
        frame2 = frame2.astype(np.float32) / 255.0
    
    blended = (1 - alpha) * frame1 + alpha * frame2
    
    return (blended * 255).astype(np.uint8)


def temporal_downsample(
    frames: Union[List[np.ndarray], torch.Tensor],
    factor: int
) -> Union[List[np.ndarray], torch.Tensor]:
    """Downsample frames temporally.
    
    Args:
        frames: Input frames
        factor: Downsample factor
        
    Returns:
        Downsampled frames
    """
    if isinstance(frames, torch.Tensor):
        T = frames.shape[0]
        indices = torch.linspace(0, T - 1, T // factor, dtype=torch.long)
        return frames[indices]
    else:
        return frames[::factor]


def temporal_upsample(
    frames: Union[List[np.ndarray], torch.Tensor],
    factor: int,
    mode: str = "linear"
) -> Union[List[np.ndarray], torch.Tensor]:
    """Upsample frames temporally using interpolation.
    
    Args:
        frames: Input frames
        factor: Upsample factor
        mode: Interpolation mode
        
    Returns:
        Upsampled frames
    """
    T = len(frames) if isinstance(frames, list) else frames.shape[0]
    
    new_T = T * factor
    
    if isinstance(frames, torch.Tensor):
        indices = torch.linspace(0, T - 1, new_T)
        
        indices_int = indices.floor().long().clamp(0, T - 1)
        indices_frac = indices.frac()
        
        frames_1 = frames[indices_int]
        frames_2 = frames[(indices_int + 1).clamp(0, T - 1)]
        
        if mode == "linear":
            result = frames_1 * (1 - indices_frac.view(-1, 1, 1, 1)) + frames_2 * indices_frac.view(-1, 1, 1, 1)
        else:
            result = frames_1
        
        return result
    else:
        indices = np.linspace(0, T - 1, new_T)
        indices_int = np.floor(indices).astype(int).clip(0, T - 1)
        indices_frac = indices - np.floor(indices)
        
        result = []
        for i in range(new_T):
            f1 = frames[indices_int[i]]
            f2 = frames[min(indices_int[i] + 1, T - 1)]
            
            if mode == "linear":
                blended = f1 * (1 - indices_frac[i]) + f2 * indices_frac[i]
            else:
                blended = f1
            
            result.append(blended.astype(np.uint8))
        
        return result


def create_video_gif(
    frames: List[np.ndarray],
    output_path: Union[str, Path],
    fps: int = 12,
    loop: int = 0
) -> None:
    """Create GIF from frames.
    
    Args:
        frames: List of frames
        output_path: Output GIF path
        fps: GIF frame rate
        loop: Loop count (0=infinite)
    """
    from PIL import Image
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    pil_frames = []
    for frame in frames:
        if frame.dtype != np.uint8:
            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        pil_frames.append(Image.fromarray(frame))
    
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000 / fps),
        loop=loop
    )