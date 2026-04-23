"""Logging utilities for Image2Video AI system."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union


class ColoredFormatter(logging.Formatter):
    """Colored log formatter for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',
        'INFO': '\033[32m',
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
        'CRITICAL': '\033[35m',
    }
    RESET = '\033[0m'
    
    def format(self, record):
        if sys.stdout.isatty():
            color = self.COLORS.get(record.levelname, self.RESET)
            record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logger(
    name: str = "Image2Video",
    level: int = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    colored: bool = True
) -> logging.Logger:
    """Setup logger with console and optional file output.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file path for logging
        colored: Use colored output for console
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.hasHandlers():
        logger.handlers.clear()
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    if colored:
        formatter = ColoredFormatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "Image2Video") -> logging.Logger:
    """Get existing logger or create new one."""
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        return setup_logger(name)
    return logger


class ProgressLogger:
    """Progress logger for tracking pipeline progress."""
    
    def __init__(
        self,
        total_steps: int,
        desc: str = "Processing",
        logger: Optional[logging.Logger] = None
    ):
        self.total_steps = total_steps
        self.current_step = 0
        self.desc = desc
        self.logger = logger or get_logger()
        self.start_time = datetime.now()
    
    def update(self, step: int = 1, message: str = "") -> None:
        """Update progress."""
        self.current_step = min(step, self.total_steps)
        progress = self.current_step / self.total_steps * 100
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        if self.current_step > 0 and self.total_steps > 0:
            eta = elapsed / self.current_step * (self.total_steps - self.current_step)
            eta_str = f"{eta:.1f}s"
        else:
            eta_str = "?"
        
        msg = f"{self.desc} [{self.current_step}/{self.total_steps}] {progress:.1f}% ETA: {eta_str}"
        if message:
            msg += f" | {message}"
        
        self.logger.info(msg)
    
    def finish(self, message: str = "Complete") -> None:
        """Mark progress as finished."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.logger.info(f"{self.desc} finished in {elapsed:.2f}s | {message}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.finish()