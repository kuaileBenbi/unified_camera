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
import sched

# from detresupdater import DetResUpdater
import cv2
import numpy as np

try:
    from .preutils import ImagePreprocessor  # 相对导入
    from .gstworker import GstPipeline
    from .detworker import DetectionWorker
    from .preworker import PreprocessWorker
    from ..ctrls.v4l2ctrlor import send_v4l2_focus_absolute_command

except ImportError:
    from preutils import ImagePreprocessor  # 直接导入（用于脚本直接运行
    from gstworker import GstPipeline
    from detworker import DetectionWorker
    from preworker import PreprocessWorker

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GLib", "2.0")
from gi.repository import Gst, GLib

Gst.init(None)

import logging

os.environ["GST_VIDEO_CONVERT_USE_RGA"] = "1"
os.environ["GST_VIDEO_FLIP_USE_RGA"] = "1"
os.environ["GST_MPP_NO_RGA"] = "0"


class ImageManager:
    def __init__(
        self,
        cam_config: dict = None,
        pipeline_config: dict = None,
        npz_path: str = None,
        bus=None,
        logger=None,
    ):

        self.pipeline_config = pipeline_config
        self.cam_config = cam_config
        self.workers = {}

        # 设置日志记录器
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)

        # 设置imager包中其他模块的日志记录器
        self._setup_imager_module_loggers()

        self.active_workers = {}

        self.queues = None

        self.shaprness_thread = None

        self.shaprness_thread = None
        self.sourcecapture_thread = None

        self.detworker = None

        self.pretool_thread = None
        self.pretool = PreprocessWorker(self.cam_config, npz_path)

        self.init_workers()
        self.detworker = DetectionWorker(self.cam_config["setting"]["tittle"])

        # 订阅曝光变化事件
        if bus is not None:
            bus.on("integration.change", self._on_integration_change)

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

    def _on_integration_change(self, t_ms: int):
        self.pretool._load_calibration(t_ms)

    def save_images(self, interval: float, num_frames: int, prefix="frame") -> bool:

        if not self.pretool.save_images(interval, num_frames, prefix):
            self.logger.warning("Opps....保存原始图像失败! ")
            return False

        return True

    def assign_detworker(self, *args) -> bool:

        if len(args) < 2:
            self.logger.debug("目标检测参数不足")
            return False

        if "v4l2_source_capturer" not in self.active_workers:
            self.logger.warning("相机未开启，目标检测图像任务无法进行！")
            return True

        detector, detection_mode, *remaining_args = args
        if not self.detworker.start(
            detector, detection_mode, self.queues, remaining_args
        ):
            return False

        self.active_workers["detect_activated"] = True

        return True

    def cancel_detworker(self) -> bool:

        if self.detworker:
            if not self.detworker.stop():
                return False

        self.logger.debug("目标检测退出成功! ")

        if "detect_activated" in self.active_workers:

            del self.active_workers["detect_activated"]

        return True

    def init_capture(self):

        if "v4l2_source_capturer" in self.active_workers:
            self.logger.debug("原始图像捕获线程已存在")
            return True

        self.queues = {
            "raw": queue.Queue(maxsize=2),  # 原始帧队列
            "enc": queue.Queue(maxsize=2),  # 编码帧队列
            "proc": queue.Queue(maxsize=2),  # 处理队列
            "enc_det": queue.Queue(maxsize=2),  # 编码队列
        }

        try:

            self.pretool_thread = threading.Thread(
                target=self.pretool.run, args=(self.queues,), daemon=True
            )
            self.pretool.running = True
            self.pretool_thread.start()

        except Exception as e:
            self.logger.error(f"启动预处理线程失败: {str(e)}")
            return False

        # 合并参数
        name = "v4l2_source_capturer"
        params = self.workers[name]["default_params"].copy()
        params.update({})

        # 创建GST实例并启动
        worker = GstPipeline(self.workers[name], self.queues)

        if not worker.start(params):
            return False

        self.active_workers[name] = worker  # 存入字典

        interval = 2.0
        fm_kwargs = {
            "sobel_ksize": 3,
            "downsample_factor": 0.5,
            "grad_threshold": 100.0,
            "roi": (160, 128, 256, 256),
        }
        try:
            # print(f"计算清晰度")
            scheduler = sched.scheduler(time.time, time.sleep)
            # 0 秒后启动第一次测量
            scheduler.enter(
                0,
                1,
                self.pretool._schedule_loop,
                (scheduler, fm_kwargs, interval),
            )
            # 用线程跑定时
            self.shaprness_thread = threading.Thread(target=scheduler.run, daemon=True)
            self.shaprness_thread.start()

        except Exception as e:
            self.logger.debug(f"开启清晰度计算错误: {str(e)}")
            return False

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

        if name == "live_encoder" and "v4l2_source_capturer" not in self.active_workers:
            if not self.init_capture():
                self.logger.error("读取相机数据失败！")
                return False

        # 合并参数
        params = self.workers[name]["default_params"].copy()
        params.update(kwargs)

        # 创建GST实例并启动
        worker = GstPipeline(self.workers[name], self.queues)

        if not worker.start(params):
            return False

        self.active_workers[name] = worker  # 存入字典

        return True

    def assign_worker(self, name: str, **kwargs) -> bool:
        """启动指定图像处理功能"""
        # print(name)
        if name not in self.workers:
            # print(name)
            self.logger.error("不存在的图像处理功能")
            return False

        # 如果已存在则先停止
        if name in self.active_workers:
            if not self.cancel_worker(name):
                return False

        if name == "live_encoder" and "v4l2_source_capturer" not in self.active_workers:
            if not self.init_capture():
                self.logger.error("读取相机数据失败！")
                return False

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

        if name not in self.active_workers:
            self.logger.warning(f"图像处理功能 {name} 未启动")
            return True  # 如果未启动则直接返回成功

        if name == "live_encoder" or name == "live_det_encoder":
            self.active_workers[name].proc_running = False
            self.logger.debug("停止推流取图")

        if not self.active_workers[name].stop():
            return False

        del self.active_workers[name]

        return True

    def cancel_all_workers(self):

        self.logger.debug("清理所有图像处理worker...")

        if not self.cancel_detworker():
            return False

        if self.pretool:
            self.logger.debug("清理预处理worker...")
            self.pretool.stop()

        if self.pretool_thread and self.pretool_thread.is_alive():
            self.pretool_thread.join()
            self.pretool_thread = None

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

    def update_preprocess_params(self, **kwargs):
        """更新预处理参数"""
        if self.pretool:
            try:
                _, invalid_keys = self.pretool.update_preprocess_params(**kwargs)
                if len(invalid_keys) > 0:
                    self.logger.error("更新预处理参数失败")
                    return False
                self.logger.info("预处理参数更新成功")
                return True
            except Exception as e:
                self.logger.error(f"更新预处理参数时发生错误: {str(e)}")
                return False
        else:
            self.logger.error("预处理工具未初始化，无法更新参数")
            return False

    def storage_blind_det_mask(self):
        if self.pretool:
            try:
                if not self.pretool.storage_blind_det_mask():
                    self.logger.error("存储坏点检测掩码失败")
                    return False
                self.logger.info("坏点检测掩码存储成功")
                return True
            except Exception as e:
                self.logger.error(f"存储坏点检测掩码时发生错误: {str(e)}")
                return False
        else:
            self.logger.error("预处理工具未初始化，无法存储坏点检测掩码")
            return False


if __name__ == "__main__":
    """
    v4l2 -> ftdout -> gst filesrc -> appsink --> pretool   -> encoding
                                             |-> detection -> encoding
    """

    import yaml, time, toml, gc, sys
    import logging

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)

    wave = sys.argv[1] if len(sys.argv) > 1 else "vis_zoom"
    pipelines_config_name = sys.argv[2] if len(sys.argv) > 1 else "pipelines_test"

    cam_config_path = f"/userdata/camera-ctrl/camera/configs/{wave}/cam_config.toml"
    with open(cam_config_path, "r", encoding="utf-8") as f:
        cam_config = toml.load(f)
        if not cam_config:
            print("摄像头配置文件加载失败")
            exit(1)

    pipelines_config_path = (
        f"/userdata/camera-ctrl/camera/configs/{wave}/{pipelines_config_name}.yaml"
    )
    with open(pipelines_config_path, "r", encoding="utf-8") as f:
        pipeline_config = yaml.safe_load(f)
        if not pipeline_config:
            print("管道配置文件加载失败")
            exit(1)

    npz_path = (
        f"/userdata/camera-ctrl/camera/infrared_correction/{wave}/{wave}_corr.yaml"
    )

    imager = ImageManager(cam_config, pipeline_config, npz_path)
    print("初始化图像管理器成功")

    # 1.测试红外with_pre: 读取摄像头原始数据 -> pretool (校正、图像翻转、十字线) & det-> 编码转发

    # v4l2管道接收到gstline -> 校正、增强、图像翻转、十字线 -> 编码转发
    if not imager.init_capture():
        print("初始化摄像头失败")
        exit(1)

    print("摄像头原始数据捕获线程已启动")

    if not imager.assign_worker(
        "live_encoder",
        bps=4000000,
        udp_clients="192.168.137.1:12345,127.0.0.1:3366,127.0.0.1:3388",
    ):
        print("编码失败")
        exit(1)

    print("编码转发线程已启动")

    module_path = "/userdata/camera-ctrl/camera"
    if module_path not in sys.path:
        sys.path.append(module_path)
        sys.path.append("/userdata/camera-ctrl")

    from ctrls.tracker import VisionTracker

    detector = VisionTracker(TPEs=3)
    detect_mode = 1
    tem_cxcywh = [0, 0, 0, 0]

    start_args = (detector, detect_mode)

    if not imager.assign_detworker(*start_args):
        print("目标检测worker启动失败")
        exit(1)

    print("目标检测worker已启动")

    if not imager.assign_worker(
        "live_det_encoder", bps=40000000, udp_clients="192.168.137.1:54321"
    ):
        print("目标检测编码转发失败")
        exit(1)

    print("目标检测编码转发线程已启动")

    loop = GLib.MainLoop()
    try:
        loop.run()
    except KeyboardInterrupt:
        print("\n停止中…")
        gc.collect()
    finally:
        self.logger.info("捕获到键盘中断，正在清理资源...")
        imager.cancel_all_workers()
        self.logger.info("所有资源已清理完毕")
