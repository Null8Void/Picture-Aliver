"""
Picture-Aliver Validation Module

Provides startup validation checks to ensure all required components
are available before running the pipeline.

Run validation:
    python -m src.picture_aliver.validate
"""

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)
logger = logging.getLogger("picture_aliver_validation")


@dataclass
class ValidationResult:
    """Result of a validation check."""
    name: str
    passed: bool
    message: str
    severity: str = "error"  # error, warning, info


class Validator:
    """Validates the Picture-Aliver installation."""
    
    def __init__(self):
        self.results: list[ValidationResult] = []
        self.project_root = Path(__file__).parent.parent
    
    def check_python_version(self) -> ValidationResult:
        """Check Python version."""
        version = sys.version_info
        passed = version.major >= 3 and version.minor >= 9
        return ValidationResult(
            name="Python Version",
            passed=passed,
            message=f"Python {version.major}.{version.minor}.{version.micro} (requires 3.9+)",
            severity="error" if not passed else "info"
        )
    
    def check_dependencies(self) -> ValidationResult:
        """Check core dependencies are installed."""
        required = {
            "torch": "torch",
            "torchvision": "torchvision", 
            "numpy": "numpy",
            "pillow": "PIL",
            "cv2": "cv2",
        }
        
        missing = []
        for name, import_name in required.items():
            try:
                __import__(import_name)
            except ImportError:
                missing.append(name)
        
        passed = len(missing) == 0
        return ValidationResult(
            name="Core Dependencies",
            passed=passed,
            message=f"Missing: {missing}" if missing else "All dependencies installed",
            severity="error" if not passed else "info"
        )
    
    def check_directories(self) -> ValidationResult:
        """Check required directories exist."""
        required_dirs = ["models", "outputs", "configs"]
        missing = []
        
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                missing.append(dir_name)
                dir_path.mkdir(exist_ok=True)
        
        return ValidationResult(
            name="Directories",
            passed=True,
            message=f"Directories ready: {required_dirs}",
            severity="info"
        )
    
    def check_gpu(self) -> ValidationResult:
        """Check GPU availability."""
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            
            if cuda_available:
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                return ValidationResult(
                    name="GPU",
                    passed=True,
                    message=f"CUDA available: {device_name} ({total_memory:.1f}GB)",
                    severity="info"
                )
            else:
                return ValidationResult(
                    name="GPU",
                    passed=True,
                    message="Running on CPU (GPU recommended for better performance)",
                    severity="warning"
                )
        except Exception as e:
            return ValidationResult(
                name="GPU",
                passed=False,
                message=f"Error checking GPU: {e}",
                severity="warning"
            )
    
    def check_imports(self) -> ValidationResult:
        """Check that pipeline can be imported."""
        try:
            # Add src to path first
            src_path = self.project_root / "src"
            if str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))
            
            # Also add project root
            if str(self.project_root) not in sys.path:
                sys.path.insert(0, str(self.project_root))
            
            from picture_aliver.main import Pipeline, PipelineConfig
            return ValidationResult(
                name="Pipeline Imports",
                passed=True,
                message="Pipeline modules imported successfully",
                severity="info"
            )
        except Exception as e:
            return ValidationResult(
                name="Pipeline Imports",
                passed=False,
                message=f"Import failed: {e}",
                severity="error"
            )
    
    def check_backend_api(self) -> ValidationResult:
        """Check API module can be loaded."""
        try:
            src_path = self.project_root / "src"
            if str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))
            
            from picture_aliver.api import app
            return ValidationResult(
                name="API Module",
                passed=True,
                message="API module loaded successfully",
                severity="info"
            )
        except Exception as e:
            return ValidationResult(
                name="API Module",
                passed=False,
                message=f"API import failed: {e}",
                severity="warning"
            )
    
    def run_all(self) -> list[ValidationResult]:
        """Run all validation checks."""
        logger.info("Running Picture-Aliver validation checks...")
        logger.info("=" * 50)
        
        checks = [
            self.check_python_version,
            self.check_dependencies,
            self.check_directories,
            self.check_gpu,
            self.check_imports,
            self.check_backend_api,
        ]
        
        for check in checks:
            result = check()
            self.results.append(result)
            
            severity_symbol = {
                "error": "[FAIL]",
                "warning": "[WARN]",
                "info": "[PASS]"
            }.get(result.severity, "[INFO]")
            
            logger.info(f"{severity_symbol} {result.name}: {result.message}")
        
        logger.info("=" * 50)
        
        errors = [r for r in self.results if r.severity == "error" and not r.passed]
        warnings = [r for r in self.results if r.severity == "warning" and not r.passed]
        
        if errors:
            logger.error(f"Validation failed with {len(errors)} error(s)")
            return self.results
        
        if warnings:
            logger.warning(f"Validation completed with {len(warnings)} warning(s)")
        else:
            logger.info("Validation passed successfully!")
        
        return self.results


def validate_early() -> bool:
    """
    Early validation - runs before app starts to catch critical issues.
    Returns True if validation passes enough to continue.
    """
    try:
        validator = Validator()
        results = validator.run_all()
        
        # Check for critical errors
        critical_failures = [r for r in results if r.severity == "error" and not r.passed]
        
        if critical_failures:
            for result in critical_failures:
                print(f"CRITICAL ERROR: {result.name} - {result.message}", file=sys.stderr)
            return False
        
        return True
        
    except Exception as e:
        print(f"Validation error: {e}", file=sys.stderr)
        return False


if __name__ == "__main__":
    validator = Validator()
    validator.run_all()