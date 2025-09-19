from datetime import datetime
import multiprocessing
import multiprocessing.queues
import os
import queue
import signal
from string import Template
import subprocess
import threading
import time

# from detresupdater import DetResUpdater
import cv2
import numpy as np

try:
    from .preutils import ImagePreprocessor  # 相对导入

    # from .gstworker import GstPipeline
    from .gstworker import GstPipeline
    from .detworker import DetectionWorker
    from .preworker import PreprocessWorker
    from ..ctrls.v4l2ctrlor import (
        send_v4l2_focus_absolute_command,
        send_v4l2_vis_command,
        send_v4l2_vis_gain_command,
    )

except ImportError:
    from preutils import ImagePreprocessor  # 直接导入（用于脚本直接运行

    # from gstworker import GstPipeline
    from gstworker import GstPipeline
    from detworker import DetectionWorker
    from preworker import PreprocessWorker

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GLib", "2.0")
from gi.repository import Gst, GLib

Gst.init(None)

import logging

import os

os.environ["GST_VIDEO_CONVERT_USE_RGA"] = "1"
os.environ["GST_VIDEO_FLIP_USE_RGA"] = "1"
os.environ["GST_MPP_NO_RGA"] = "0"


class ImageManager:
    def __init__(
        self, cam_config: dict, pipeline_config: dict, npz_path: str, logger=None
    ):
        self.camera_identity = self.cam_config["setting"]["tittle"]

        self.pipeline_config = pipeline_config
        self.cam_config = cam_config
        self.workers = {}
        self.active_workers = {}

        # 设置日志记录器
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)

        # 设置imager包中其他模块的日志记录器
        self._setup_imager_module_loggers()

        self.queues = None

        self.shaprness_thread = None
        self.sourcecapture_thread = None

        self.detworker = None

        self.init_workers()
        self.detworker = DetectionWorker()

    def _setup_imager_module_loggers(self):
        """设置imager包中其他模块的日志记录器"""
        try:
            # 设置detworker模块的日志记录器
            from .detworker import set_detworker_logger

            set_detworker_logger(self.logger)

            # 设置gstworker模块的日志记录器
            from .gstworker import set_gstworker_logger

            set_gstworker_logger(self.logger)

            # 设置preutils模块的日志记录器
            from .preutils import set_preutils_logger

            set_preutils_logger(self.logger)

            # 设置preworker模块的日志记录器
            from .preworker import set_preworker_logger

            set_preworker_logger(self.logger)

            # 设置updater模块的日志记录器
            from .updater import set_updater_logger

            set_updater_logger(self.logger)

            self.logger.debug("imager包模块日志记录器设置完成")
        except Exception as e:
            self.logger.warning(f"设置imager包模块日志记录器失败: {e}")

    def assign_detworker(self, *args) -> bool:

        if len(args) < 2:
            self.logger.debug("目标检测参数不足")
            return False

        detector, detection_mode, *remaining_args = args
        if not self.detworker.start_vis(
            detector, detection_mode, self.queues, remaining_args
        ):
            return False

        return True

    def cancel_detworker(self) -> bool:

        if self.detworker:
            if not self.detworker.stop_vis():
                return False

        self.logger.debug("目标检测退出成功! ")

        return True

    def init_workers(self):
        """初始化所有模板"""
        for name, spec in self.pipeline_config["pipelines"].items():
            self.workers[name] = {
                "type": spec["type"],
                "template": spec["template"],
                "default_params": spec["params"],
                "description": spec["description"],
            }

    def assign_switch_worker(self, name: str, **kwargs) -> bool:
        """启动指定图像处理功能"""
        # print(name)
        if name not in self.workers:
            # print(name)
            self.logger.error("不存在的图像处理功能")
            return False

        # 如果已存在则先停止
        if name in self.active_workers:
            return True

        # 合并参数
        params = self.workers[name]["default_params"].copy()
        params.update(kwargs)

        # 创建GST实例并启动
        worker = GstPipeline(self.workers[name], self.queues)

        if not worker.start(params):
            return False

        self.active_workers[name] = worker  # 存入字典

        time.sleep(3)

        if self.camera_identity in ["vis_zoom", "vis_fix"] and name == "live_encoder":
            send_v4l2_vis_gain_command(params["device"])

        return True

    def assign_worker(self, name: str, **kwargs) -> bool:
        """启动指定图像处理功能"""
        # print(name)
        if name not in self.workers:
            self.logger.error(f"不存在的图像处理功能: {name}")
            return False

        # 如果已存在则先停止
        if name in self.active_workers:
            if not self.cancel_worker(name):
                return False

        if name == "live_det_encoder":
            if self.queues is None:
                # 多次重启时，不重建，保证在detectworker中清空队列
                self.queues = {"det": queue.Queue(maxsize=2), "det_res": {}}

        if name == "live_encoder" and "live_det_encoder" in self.active_workers:
            # 如果正在目标检测的时候修改编码码率
            # 就直接修改目标检测流里的推流参数
            name = "live_det_encoder"

        # 合并参数
        params = self.workers[name]["default_params"].copy()
        params.update(kwargs)

        # 创建GST实例并启动
        worker = GstPipeline(self.workers[name], self.queues)

        if not worker.start(params):
            return False

        self.active_workers[name] = worker  # 存入字典

        return True

    def cancel_worker(self, name: str) -> bool:
        """关闭指定流水线"""
        if name not in self.workers:
            self.logger.error("不存在的图像处理功能")
            return False

        if name in self.active_workers:
            if not self.active_workers[name].stop():
                return False
            del self.active_workers[name]

        return True

    def cancel_all_workers(self):

        self.logger.debug("清理所有图像处理worker...")

        if not self.cancel_detworker():
            return False

        for name in list(self.active_workers.keys()):

            self.logger.debug(f"开始取消gstreamer命令: {name}")

            if not self.cancel_worker(name):
                return False

        if self.queues is not None:
            for _, q in self.queues.items():
                while not q.empty():
                    q.get_nowait()

        self.logger.debug("所有图像处理worker已清理完毕")

        return True
