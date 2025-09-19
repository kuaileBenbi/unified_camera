"""
统一相机控制包
支持多种相机模式：lwir_fix, mwir_fix, mwir_zoom, swir_fix, vis_fix, vis_zoom
启动方式：
import unified_camera

# 启动相机
unified_camera.start(["vis_fix", "mwir_fix"], log_level="INFO", log_dir="logs")
# 停止相机
unified_camera.stop(["vis_fix", "mwir_fix"])
# 检查运行状态
unified_camera.is_running("vis_fix")
# 列出所有运行的相机
unified_camera.list_running()

########################################################
# 检查运行状态
if unified_camera.is_running("swir_fix"):
    print("swir_fix 正在运行")
    pid = unified_camera.get_camera_pid("swir_fix")
    print(f"swir_fix PID: {pid}")
"""

import multiprocessing
import sys
from pathlib import Path


__version__ = "1.0.0"
__author__ = "Camera Control@GOODLUCK"

# 全局进程字典
_camera_processes = {}

try:
    from .core.unified_camera import UnifiedCameraController
    from .config.config_manager import ConfigManager
    from .utils.logger import setup_logger

    def start(modes, config_path="configs", log_level="INFO", log_dir="logs"):
        """
        启动相机进程
        
        Args:
            modes: 相机模式列表，如 ["vis_fix", "mwir_fix"]
            config_path: 配置文件路径
            log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: 日志文件目录
        """
        if not isinstance(modes, list):
            modes = [modes]
        
        for mode in modes:
            if mode in _camera_processes and _camera_processes[mode].is_alive():
                print(f"相机 {mode} 已经在运行")
                continue
            
            def worker():
                """工作进程函数"""
                try:
                    # 设置日志
                    setup_logger(mode, log_level, log_dir)
                    
                    config_manager = ConfigManager(config_path)
                    controller = UnifiedCameraController(mode, config_manager)
                    controller.process()
                except Exception as e:
                    print(f"相机 {mode} 启动失败: {e}")
                    sys.exit(1)
            
            # 创建并启动进程
            process = multiprocessing.Process(target=worker, name=f"camera-{mode}")
            process.start()
            _camera_processes[mode] = process
            
            print(f"相机 {mode} 已启动，PID: {process.pid}")
    
    def stop(modes=None):
        """
        停止相机进程
        
        Args:
            modes: 相机模式列表，如果为None则停止所有相机
        """
        if modes is None:
            modes = list(_camera_processes.keys())
        elif not isinstance(modes, list):
            modes = [modes]
        
        for mode in modes:
            if mode in _camera_processes:
                process = _camera_processes[mode]
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
                del _camera_processes[mode]
                print(f"相机 {mode} 已停止")
    
    def is_running(mode):
        """检查相机是否运行"""
        return mode in _camera_processes and _camera_processes[mode].is_alive()
    
    def list_running():
        """列出所有运行的相机"""
        running = {}
        for mode, process in _camera_processes.items():
            if process.is_alive():
                running[mode] = process.pid
        return running

    __all__ = [
        "UnifiedCameraController",
        "ConfigManager", 
        "setup_logger",
        "start",
        "stop", 
        "is_running",
        "list_running"
    ]
except ImportError as e:
    print(f"导入错误: {e}")
    __all__ = []
