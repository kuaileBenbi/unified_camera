import logging
import os
from pathlib import Path
from typing import Optional


def setup_logger(
    mode: str, 
    log_level: str = "INFO",
    log_dir: str = "logs",
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    为指定模式设置独立的日志记录器
    
    Args:
        mode: 相机模式 (lwir_fix, mwir_fix, mwir_zoom, swir_fix, vis_fix, vis_zoom)
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: 日志文件目录
        log_format: 日志格式，如果为None则使用默认格式
        
    Returns:
        配置好的日志记录器
    """
    # 创建日志目录
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # 设置日志文件名
    log_file = log_path / f"{mode}.log"
    
    # 创建日志记录器
    logger = logging.getLogger(f"unified_camera.{mode}")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 清除已有的处理器
    logger.handlers.clear()
    
    # 设置日志格式
    if log_format is None:
        log_format = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(filename)s:%(lineno)d - %(message)s"
        )
    
    formatter = logging.Formatter(log_format)
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 防止日志重复
    logger.propagate = False
    
    return logger


def get_logger(mode: str) -> logging.Logger:
    """
    获取指定模式的日志记录器
    
    Args:
        mode: 相机模式
        
    Returns:
        日志记录器
    """
    return logging.getLogger(f"unified_camera.{mode}")


def setup_root_logger(log_level: str = "INFO", log_dir: str = "logs"):
    """
    设置根日志记录器
    
    Args:
        log_level: 日志级别
        log_dir: 日志文件目录
    """
    # 创建日志目录
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # 设置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # 清除已有的处理器
    root_logger.handlers.clear()
    
    # 设置日志格式
    log_format = (
        "%(asctime)s - %(name)s - %(levelname)s - "
        "%(filename)s:%(lineno)d - %(message)s"
    )
    formatter = logging.Formatter(log_format)
    
    # 文件处理器
    log_file = log_path / "unified_camera.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
