"""
Picture-Aliver Mobile - Local Backend Server

This module allows the mobile app to run a local backend server
directly on the device, enabling fully offline operation.

For the bundled APK, this server runs alongside the React Native app.

Usage:
    from mobile_backend import MobileBackend
    backend = MobileBackend(port=8000)
    backend.start()
"""

import os
import sys
import time
import threading
import logging
import socket
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger('mobile_backend')


class MobileBackend:
    """
    Local backend server for mobile APK.
    
    Runs the FastAPI pipeline in a background thread on the mobile device,
    enabling fully standalone operation without WiFi.
    """
    
    def __init__(self, port: int = 8000):
        self.port = port
        self.server_thread: Optional[threading.Thread] = None
        self.running = False
        self.app = None
        
    def check_port_available(self, port: int) -> bool:
        """Check if a port is available."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return True
        except OSError:
            return False
    
    def find_available_port(self, start_port: int = 8000) -> int:
        """Find an available port starting from start_port."""
        port = start_port
        max_attempts = 100
        for _ in range(max_attempts):
            if self.check_port_available(port):
                return port
            port += 1
        raise RuntimeError(f"Could not find available port in range {start_port}-{start_port + max_attempts}")
    
    def start(self, port: Optional[int] = None) -> str:
        """
        Start the backend server.
        
        Returns:
            The URL where the server is running (e.g., "http://127.0.0.1:8000")
        """
        if self.running:
            return f"http://127.0.0.1:{self.port}"
        
        if port is None:
            port = self.port
        
        # Find available port if requested port is in use
        if not self.check_port_available(port):
            logger.warning(f"Port {port} is in use, finding available port...")
            port = self.find_available_port(port)
        
        self.port = port
        
        try:
            # Start server in background thread
            def run_server():
                import uvicorn
                from src.picture_aliver.api import app
                
                logger.info(f"Starting mobile backend on port {port}...")
                
                uvicorn.run(
                    app,
                    host="127.0.0.1",
                    port=port,
                    log_level="warning",
                    access_log=False
                )
            
            self.server_thread = threading.Thread(target=run_server, daemon=True)
            self.server_thread.start()
            self.running = True
            
            # Wait for server to start
            time.sleep(2)
            
            url = f"http://127.0.0.1:{port}"
            logger.info(f"Mobile backend started at {url}")
            
            return url
            
        except Exception as e:
            logger.error(f"Failed to start mobile backend: {e}")
            raise
    
    def stop(self):
        """Stop the backend server."""
        if self.running:
            self.running = False
            logger.info("Mobile backend stopped")
    
    def is_running(self) -> bool:
        """Check if backend is running."""
        return self.running
    
    def get_status(self) -> dict:
        """Get backend status."""
        return {
            "running": self.running,
            "port": self.port,
            "url": f"http://127.0.0.1:{self.port}" if self.running else None
        }


# Global backend instance
_backend: Optional[MobileBackend] = None


def get_backend() -> MobileBackend:
    """Get or create the global backend instance."""
    global _backend
    if _backend is None:
        _backend = MobileBackend()
    return _backend


def start_mobile_backend(port: Optional[int] = None) -> str:
    """Convenience function to start the mobile backend."""
    return get_backend().start(port)


def stop_mobile_backend():
    """Convenience function to stop the mobile backend."""
    get_backend().stop()