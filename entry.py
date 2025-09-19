#!/usr/bin/env python3
"""
统一相机控制主入口
支持多种相机模式：lwir_fix, mwir_fix, mwir_zoom, swir_fix, vis_fix, vis_zoom
"""

import argparse
import sys
import os
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from core.unified_camera import UnifiedCameraController
from config.config_manager import ConfigManager
from utils.logger import setup_logger, setup_root_logger


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="统一相机控制器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
支持的相机模式:
  lwir_fix    - 长波红外固定焦距相机
  mwir_fix    - 中波红外固定焦距相机  
  mwir_zoom   - 中波红外变焦相机
  swir_fix    - 短波红外固定焦距相机
  vis_fix     - 可见光固定焦距相机
  vis_zoom    - 可见光变焦相机

使用示例:
  python main.py --mode lwir_fix
  python main.py --mode mwir_zoom --log-level DEBUG
  python main.py --mode vis_fix --config-path ./configs
        """,
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "lwir_fix",
            "mwir_fix",
            "mwir_zoom",
            "swir_fix",
            "vis_fix",
            "vis_zoom",
        ],
        help="相机模式",
    )

    parser.add_argument(
        "--config-path", default="configs", help="配置文件路径 (默认: configs)"
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="日志级别 (默认: INFO)",
    )

    parser.add_argument("--log-dir", default="logs", help="日志文件目录 (默认: logs)")

    parser.add_argument("--version", action="version", version="统一相机控制器 v1.0.0")

    return parser.parse_args()


def validate_config_path(config_path: str) -> bool:
    """验证配置文件路径"""
    config_dir = Path(config_path)
    if not config_dir.exists():
        print(f"错误: 配置文件目录不存在: {config_path}")
        return False

    # 检查是否有必要的配置文件
    required_files = ["cam_config.toml", "pipelines.yaml"]
    for mode in [
        "lwir_fix",
        "mwir_fix",
        "mwir_zoom",
        "swir_fix",
        "vis_fix",
        "vis_zoom",
    ]:
        mode_dir = config_dir / mode
        if not mode_dir.exists():
            print(f"警告: 模式 {mode} 的配置目录不存在: {mode_dir}")
            continue

        for file in required_files:
            if not (mode_dir / file).exists():
                print(f"警告: 模式 {mode} 缺少配置文件: {file}")

    return True


def main():
    """主函数"""
    try:
        # 解析命令行参数
        args = parse_arguments()

        # 验证配置文件路径
        if not validate_config_path(args.config_path):
            sys.exit(1)

        # 设置根日志记录器
        setup_root_logger(args.log_level, args.log_dir)

        # 设置模式特定的日志记录器
        logger = setup_logger(args.mode, args.log_level, args.log_dir)

        logger.info(f"启动统一相机控制器 - 模式: {args.mode}")
        logger.info(f"配置文件路径: {args.config_path}")
        logger.info(f"日志级别: {args.log_level}")
        logger.info(f"日志目录: {args.log_dir}")

        # 初始化配置管理器
        config_manager = ConfigManager(args.config_path)

        # 验证配置
        try:
            cam_config = config_manager.get_camera_config(args.mode)
            imager_config = config_manager.get_imager_config(args.mode)
            logger.info(f"成功加载 {args.mode} 配置")
        except Exception as e:
            logger.error(f"加载 {args.mode} 配置失败: {e}")
            sys.exit(1)

        # 创建并启动相机控制器
        controller = UnifiedCameraController(args.mode, config_manager)

        logger.info(f"相机控制器初始化完成，开始运行...")
        controller.process()

    except KeyboardInterrupt:
        print("\n收到中断信号，正在退出...")
        sys.exit(0)
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
