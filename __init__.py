"""
统一相机控制包
支持多种相机模式：lwir_fix, mwir_fix, mwir_zoom, swir_fix, vis_fix, vis_zoom
"""

__version__ = "1.0.0"
__author__ = "Camera Control Team"

try:
    from .core.unified_camera import UnifiedCameraController
    from .config.config_manager import ConfigManager
    from .utils.logger import setup_logger

    __all__ = [
        "UnifiedCameraController",
        "ConfigManager", 
        "setup_logger"
    ]
except ImportError as e:
    print(f"导入错误: {e}")
    __all__ = []
