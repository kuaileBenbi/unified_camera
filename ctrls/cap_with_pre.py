import asyncio
from datetime import datetime
import logging
import os
from pathlib import Path
import threading
import time
from typing import List
import aiofiles
from tomlkit import parse, dumps, table
from status import get_values
from .v4l2ctrlor import v4l2_cmd

from ..utils import (
    estimated_size_h264,
    estimated_image_size,
    measure_map_once,
    measure_map_sched,
)

storage_path_map = {
    "/mnt/disk0": "storage_space_remaining1",
    "/mnt/disk1": "storage_space_remaining2",
}

class CapCtrl:

    def __init__(self, cam_config, imager_manager, cam_ctrl=None, logger=None):
        self.config = cam_config
        self.imager_manager = imager_manager
        self.cam_ctrl = cam_ctrl
        self.query_task_t: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.camera_tittle = self.config["setting"]["tittle"]
        
        # 设置日志记录器
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
    
    # ---------- 内部：查询任务 ----------
    def _measure_loop(
        self, once_ids, sched_ids, interval=10, max_retries=5, retry_sleep=3
    ):
        """
        后台线程入口：先做一次性查询（带重试），全部成功后进入周期性查询。
        注意：cam_dev.v4l2_r 是 async，这里在线程内建立并复用一个 event loop。
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        def run_awaitable(awaitable):
            return loop.run_until_complete(awaitable)

        # 1) 一次性查询：曝光/增益等
        for pid in once_ids:
            ok = False
            for attempt in range(1, max_retries + 1):
                if self.stop_event.is_set():
                    self.logger.info("[measure] 收到停止信号，结束一次性查询")
                    loop.close()
                    return
                try:
                    ok = run_awaitable(self.cam_ctrl.v4l2_r(v4l2_cmd[pid]))
                except Exception:
                    self.logger.exception(f"[measure] v4l2_r({pid}) 异常（第{attempt}次）")
                    ok = False
                if ok:
                    break
                time.sleep(retry_sleep)
            if not ok:
                self.logger.error(
                    f"[measure] 一次性查询 {pid} 最多重试{max_retries}次仍失败，终止状态检测"
                )
                loop.close()
                return

        self.logger.info("[measure] 一次性查询全部成功，进入周期性查询")

        # 2) 周期性查询：对焦/温度/电流等
        while not self.stop_event.is_set():
            for pid in sched_ids:
                try:
                    ok = run_awaitable(self.cam_ctrl.v4l2_r(v4l2_cmd[pid]))
                except Exception:
                    self.logger.exception(
                        f"[measure] 周期查询 v4l2_r({pid}) 异常，停止状态检测"
                    )
                    loop.close()
                    return
                if not ok:
                    self.logger.error(f"[measure] 周期查询 {pid} 失败，停止状态检测")
                    loop.close()
                    return
            # 可被 stop_event 立刻打断的等待
            self.stop_event.wait(interval)

        loop.close()

    def _start_query_task_if_needed(self, once_ids, sched_ids, interval):
        if self.query_task_t and self.query_task_t.is_alive():
            self.logger.info("[measure] 查询任务已在运行，忽略重复启动")
            return
        self.stop_event.clear()
        self.query_task_t = threading.Thread(
            target=self._measure_loop,
            args=(once_ids, sched_ids, interval),
            daemon=True,
        )
        self.query_task_t.start()
        self.logger.info("[measure] 查询任务已启动")

    def _stop_query_task(self):
        if self.query_task_t:
            self.stop_event.set()
            self.query_task_t.join(timeout=5)
            self.query_task_t = None
            self.logger.info("[measure] 查询任务已停止")

    def set_addr(self, address_ip: str, address_port: int) -> bool:
        # 0x11
        self.logger.debug(f"网络接口设为: {address_ip, address_port}")
        self.config["setting"]["encoder_params"]["udp_ip"] = address_ip
        self.config["setting"]["encoder_params"]["udp_port"] = address_port
        return True

    def set_encode_switch(self, encoder_switch: int) -> bool:
        # 0x29
        self.logger.debug(f"实时视频开关: {encoder_switch}")

        video_location = "/dev/null"
        image_location = "/dev/null"

        if encoder_switch == 0:
            self._stop_query_task()
            # 停止视频流编码
            if not self.imager_manager.cancel_all_workers():
                return False
            return True

        elif encoder_switch == 1:
            # 开启视频流编码

            if self.config["setting"]["storage_params"]["video_name"] == "/dev/null":

                video_location = self.config["setting"]["storage_params"]["video_name"]
                video_max_files = 0

            else:

                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                output_path_video = os.path.join(
                    self.config["setting"]["storage_params"]["video_path"],
                    "auto_save",
                    "video",
                    timestamp,
                )
                os.makedirs(output_path_video, exist_ok=True)

                dur = 10  # 默认分段时长为10秒

                # 计算能存储的文件数量
                parent_path = self.config["setting"]["storage_params"]["video_path"]
                mount_point = str(Path(parent_path).parents[1])
                video_max_files = self.compute_free_space(mount_point, "video", dur)

                if video_max_files <= 0:
                    video_max_files = 0

                video_location = f"{output_path_video}/{self.config["setting"]["storage_params"]["video_name"]}"

            if self.config["setting"]["storage_params"]["image_name"] == "/dev/null":

                image_location = self.config["setting"]["storage_params"]["image_name"]
                image_max_files = 0
            else:

                output_path_image = os.path.join(
                    self.config["setting"]["storage_params"]["image_path"],
                    "auto_save",
                    "image",
                    timestamp,
                )
                os.makedirs(output_path_image, exist_ok=True)

                # 计算能存储的文件数量
                parent_path = self.config["setting"]["storage_params"]["image_path"]
                mount_point = str(Path(parent_path).parents[1])
                image_max_files = self.compute_free_space(mount_point, "image")

                if image_max_files <= 0:
                    image_max_files = 0

                image_location = f"{output_path_image}/{self.config["setting"]["storage_params"]["image_name"]}"

            params = {
                "video_max_files": video_max_files,
                "video_location": video_location,
                "image_max_files": image_max_files,
                "image_location": image_location,
                "bps": self.config["setting"]["encoder_params"]["bps"],
                "udp_clients": f"""{self.config["setting"]["encoder_params"]["udp_ip"]}:{self.config["setting"]["encoder_params"]["udp_port"]},{self.config["setting"]["storage_params"]["video_ip"]}:{self.config["setting"]["storage_params"]["video_port"]},{self.config["setting"]["storage_params"]["image_ip"]}:{self.config["setting"]["storage_params"]["image_port"]}""",
            }

            if not self.imager_manager.assign_switch_worker("live_encoder", **params):
                self.logger.error("启动失败")
                return False
            
            if not self.imager_manager.assign_worker("live_det_encoder"):
                logging.error("预启动目标检测失败！")
                return False
            # 等待推流稳定后，再启动状态查询
            time.sleep(2)
            once_ids = measure_map_once[self.camera_tittle]
            sched_ids = measure_map_sched[self.camera_tittle]
            self._start_query_task_if_needed(once_ids, sched_ids, interval=2)

            return True
        else:
            self.logger.warning("不存在的视频流参数设置!")
            return False

    def set_encode_setting(self, kbps: int, fps: int, scale=True) -> bool:
        # 0x20
        self.logger.debug(f"视频流设置为: {kbps, fps}")

        video_location = "/dev/null"
        image_location = "/dev/null"

        if kbps == 0 or fps == 0:
            self.logger.warning("帧频或码率设置值为0! 视频流设置失败!")
            return False

        if kbps > 20000:
            kbps = 20000

        if scale:
            # 码率缩放
            kbps = int(kbps * 0.6)

        self.config["setting"]["encoder_params"]["bps"] = int(kbps * 1000)
        self.config["setting"]["encoder_params"]["fps"] = fps

        if self.config["setting"]["storage_params"]["video_name"] == "/dev/null":

            video_location = self.config["setting"]["storage_params"]["video_name"]
            video_max_files = 0
        else:

            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            output_path_video = os.path.join(
                self.config["setting"]["storage_params"]["video_path"],
                "auto_save",
                "video",
                timestamp,
            )
            os.makedirs(output_path_video, exist_ok=True)

            dur = 10  # 默认分段时长为10秒

            # 计算能存储的文件数量
            parent_path = self.config["setting"]["storage_params"]["video_path"]
            mount_point = str(Path(parent_path).parents[1])
            video_max_files = self.compute_free_space(mount_point, "video", dur)
            if video_max_files <= 0:
                video_max_files = 0

            video_location = f"{output_path_video}/{self.config["setting"]["storage_params"]["video_name"]}"

        if self.config["setting"]["storage_params"]["image_name"] == "/dev/null":

            image_location = self.config["setting"]["storage_params"]["image_name"]
            image_max_files = 0
        else:

            output_path_image = os.path.join(
                self.config["setting"]["storage_params"]["image_path"],
                "auto_save",
                "image",
                timestamp,
            )
            os.makedirs(output_path_image, exist_ok=True)

            # 计算能存储的文件数量
            parent_path = self.config["setting"]["storage_params"]["image_path"]
            mount_point = str(Path(parent_path).parents[1])
            image_max_files = self.compute_free_space(mount_point, "image")

            if image_max_files <= 0:
                image_max_files = 0

            image_location = f"{output_path_image}/{self.config["setting"]["storage_params"]["image_name"]}"

        params = {
            "video_max_files": video_max_files,
            "video_location": video_location,
            "image_max_files": image_max_files,
            "image_location": image_location,
            "bps": self.config["setting"]["encoder_params"]["bps"],
            "udp_clients": f"""{self.config["setting"]["encoder_params"]["udp_ip"]}:{self.config["setting"]["encoder_params"]["udp_port"]},{self.config["setting"]["storage_params"]["video_ip"]}:{self.config["setting"]["storage_params"]["video_port"]},{self.config["setting"]["storage_params"]["image_ip"]}:{self.config["setting"]["storage_params"]["image_port"]}""",
        }

        if not self.imager_manager.assign_worker("live_encoder", **params):
            self.logger.error("启动失败")
            return False

        return True

    def compute_free_space(self, storage_path, imgorvideo, dur=1) -> int:
        """计算剩余存储空间"""

        # 1) 读空闲空间 -> 兆
        free_bytes = get_values(storage_path_map[storage_path])[0]

        # 2) 估算每个文件大小 -> 兆
        if imgorvideo == "video":
            bps = self.config["setting"]["encoder_params"]["bps"]
            estimated_per_file = estimated_size_h264(
                bitrate_bps=bps, segment_duration_s=dur
            )
        elif imgorvideo == "image":
            w = self.config["setting"]["image_width"]
            h = self.config["setting"]["image_height"]
            estimated_per_file = estimated_image_size(w=w, h=h)

        max_files = int((free_bytes / estimated_per_file) * 0.85 * 0.5 * (1 / 6))
        if max_files <= 0:
            self.logger.warning("存储空间不足, 无法存储!")
            return 0

        return max_files

    def set_video_storge(self, video_storage_switch: int, dur: int = None) -> bool:
        # 0x21

        self.logger.debug(f"视频存储开关: {video_storage_switch}")

        if video_storage_switch == 0:

            if self.config["setting"]["storage_params"]["video_name"] == "/dev/null":
                self.logger.warning("视频存储已关闭, 无需重复操作!")
                return True

            # 关闭视频存储
            self.config["setting"]["storage_params"]["video_name"] = "/dev/null"
            self.config["setting"]["storage_params"]["image_name"] = "/dev/null"

            if not self.imager_manager.cancel_worker("storage_splitvideo"):
                return False

        elif video_storage_switch == 1:

            if (
                self.config["setting"]["storage_params"]["video_name"]
                == "video_%Y%m%d_%H%M%S.h264"
            ):
                self.logger.warning("视频存储已开启, 无需重复操作!")
                return True

            # 开启视频存储
            self.config["setting"]["storage_params"][
                "video_name"
            ] = "video_%Y%m%d_%H%M%S.h264"
            self.config["setting"]["storage_params"][
                "image_name"
            ] = "image_%Y%m%d_%H%M%S.jpg"

        else:
            self.logger.warning("不存在的视频存储参数设置!")
            return False

        if self.config["setting"]["storage_params"]["video_name"] == "/dev/null":

            video_location = self.config["setting"]["storage_params"]["video_name"]
            video_max_files = 0
        else:

            # 更新配置
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            output_path_video = os.path.join(
                self.config["setting"]["storage_params"]["video_path"],
                "auto_save",
                "video",
                timestamp,
            )
            os.makedirs(output_path_video, exist_ok=True)

            dur = 10  # 默认分段时长为10秒

            # 计算能存储的文件数量
            parent_path = self.config["setting"]["storage_params"]["video_path"]
            mount_point = str(Path(parent_path).parents[1])
            video_max_files = self.compute_free_space(mount_point, "video", dur)
            if video_max_files <= 0:
                video_max_files = 0

            video_location = f"{output_path_video}/{self.config["setting"]["storage_params"]["video_name"]}"

        if self.config["setting"]["storage_params"]["image_name"] == "/dev/null":

            image_location = self.config["setting"]["storage_params"]["image_name"]
            image_max_files = 0
        else:

            output_path_image = os.path.join(
                self.config["setting"]["storage_params"]["image_path"],
                "auto_save",
                "image",
                timestamp,
            )
            os.makedirs(output_path_image, exist_ok=True)

            # 计算能存储的文件数量
            parent_path = self.config["setting"]["storage_params"]["image_path"]
            mount_point = str(Path(parent_path).parents[1])
            image_max_files = self.compute_free_space(mount_point, "image")

            if image_max_files <= 0:
                image_max_files = 0

            image_location = f"{output_path_image}/{self.config["setting"]["storage_params"]["image_name"]}"

        params = {
            "video_max_files": video_max_files,
            "video_location": video_location,
            "image_max_files": image_max_files,
            "image_location": image_location,
            "bps": self.config["setting"]["encoder_params"]["bps"],
            "udp_clients": f"""{self.config["setting"]["encoder_params"]["udp_ip"]}:{self.config["setting"]["encoder_params"]["udp_port"]},{self.config["setting"]["storage_params"]["video_ip"]}:{self.config["setting"]["storage_params"]["video_port"]},{self.config["setting"]["storage_params"]["image_ip"]}:{self.config["setting"]["storage_params"]["image_port"]}""",
        }

        if not self.imager_manager.assign_switch_worker("live_encoder", **params):
            self.logger.error("启动失败")
            return False

        return True

    def set_video_storge_setting(self, kbps: int, fps: int, dur: int) -> bool:
        # 0x22
        self.logger.debug(f"视频存储设置: {kbps, fps, dur}")

        # 开启视频存储
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_path = os.path.join(
            self.config["setting"]["storage_params"]["video_path"],
            "manual_save",
            timestamp,
        )
        os.makedirs(output_path, exist_ok=True)

        self.config["setting"]["storage_params"]["video_seg"] = dur

        # 计算能存储的文件数量
        parent_path = self.config["setting"]["storage_params"]["video_path"]
        mount_point = str(Path(parent_path).parents[1])
        max_files = self.compute_free_space(mount_point, "video", dur)
        if max_files <= 0:
            self.logger.warning("存储空间不足, 无法存储视频!")
            return False
        max_files_to_use = min(5, max_files)
        params = {
            "segment_duration": int(dur * 1000000000),
            "max_files": max_files_to_use,
            "output_path": f"{output_path}/video_%Y%m%d_%H%M%S.mkv",
        }

        if not self.imager_manager.assign_worker("storage_splitvideo", **params):
            return False

        # 自动取消逻辑
        duration = dur * max_files_to_use
        self.logger.info(f"手动视频存储预计持续 {duration}s，将在时间到后自动关闭")

        # 若上一次定时器未取消，先取消
        if hasattr(self, "_video_timer") and self._video_timer:
            self._video_timer.cancel()

        def auto_cancel_video():
            self.logger.info("视频录制时间到，自动关闭视频存储")
            try:
                if not self.imager_manager.cancel_worker("storage_splitvideo"):
                    self.logger.error("自动取消视频存储失败")
            except Exception as e:
                self.logger.error(f"自动取消视频存储异常: {e}")
            finally:
                if hasattr(self, "_video_timer") and self._video_timer:
                    self._video_timer.cancel()

            self._video_timer = None

        self._video_timer = threading.Timer(duration, auto_cancel_video)
        self._video_timer.daemon = True
        self._video_timer.start()

        return True

    def set_image_storage(
        self, image_storage_switch: int, inter: int, max_files: int
    ) -> bool:
        self.logger.debug(f"自动存图开关: {image_storage_switch, inter, max_files}")
        inter_sec = inter * 0.1  # LSB = 0.1s

        if image_storage_switch == 0:
            # 关闭图片存储，不算失败，直接返回 True
            return True

        elif image_storage_switch == 1:
            # 开启图片存储

            if inter_sec == 0:
                self.logger.warning("存图间隔设置值为0! 图片存储失败!")
                return False

            if not self.imager_manager.save_images(
                interval=inter_sec, num_frames=max_files, prefix="raw"
            ):
                self.logger.warning("存图失败！")

                return False

        return True

    def set_image_storage_GST(
        self, image_storage_switch: int, inter: int, max_files: int
    ) -> bool:
        # 0x23
        self.logger.debug(f"自动存图开关: {image_storage_switch, inter}")
        inter_sec = inter * 0.1  # LSB = 0.1s

        if image_storage_switch == 0:
            # 关闭图片存储
            if not self.imager_manager.cancel_worker("storage_singleframe"):
                return False
            return True

        elif image_storage_switch == 1:
            # 开启图片存储

            if inter_sec == 0:
                self.logger.warning("存图间隔设置值为0! 图片存储失败!")
                return False

            try:
                inv = 1 / inter_sec
                if inv.is_integer():
                    fps_num = int(inv)
                    fps_den = 1
                else:
                    fps_num = 1
                    fps_den = round(inter_sec, 3)
                fps = f"{int(fps_num)}/{int(fps_den)}"
            except ZeroDivisionError:
                self.logger.warning("存图间隔不能为0!")
                return False

            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            output_path = os.path.join(
                self.config["setting"]["storage_params"]["image_path"],
                "manual_save",
                timestamp,
            )
            os.makedirs(output_path, exist_ok=True)

            # 计算能存储的文件数量
            parent_path = self.config["setting"]["storage_params"]["image_path"]
            mount_point = str(Path(parent_path).parents[1])
            max_files_free = self.compute_free_space(mount_point, "image")

            if max_files > max_files_free:
                self.logger.warning("存储空间不足, 无法存储图像!")
                return False

            params = {
                "fps": fps,
                "max_files": max_files,
                "output_path": f"{output_path}/image_%Y%m%d_%H%M%S.jpg",
            }

            if not self.imager_manager.assign_worker("storage_singleframe", **params):
                return False

            # 自动定时取消逻辑
            duration = inter_sec * max_files
            self.logger.info(f"预计存图时长: {duration:.2f}s, 设置定时器自动停止")

            # 若上一次定时器未取消，先取消
            if hasattr(self, "_image_timer") and self._image_timer:
                self._image_timer.cancel()

            def auto_cancel():
                self.logger.info("自动存图时间到，执行关闭")
                try:
                    if not self.imager_manager.cancel_worker("storage_singleframe"):
                        self.logger.error("自动取消存图失败")
                except Exception as e:
                    self.logger.error(f"自动取消存图异常: {e}")
                finally:
                    if hasattr(self, "_image_timer") and self._video_timer:
                        self._image_timer.cancel()

            self._image_timer = threading.Timer(duration, auto_cancel)
            self._image_timer.daemon = True
            self._image_timer.start()

            return True

        else:
            self.logger.warning("不存在的自动存图参数设置!")
            return False

    def set_storge_setting(self, video_path: int, image_path: int) -> bool:
        # 0x25

        self.logger.debug(f"视频存储设置路径: {video_path} 图像存储设置路径: {image_path}")

        if video_path not in (0, 1) or image_path not in (0, 1):
            self.logger.warning("存储路径设置参数不合法")
            return False

        video_storage_dir = (
            self.config["setting"]["ssd"]["ssd1"]
            if video_path == 0
            else self.config["setting"]["ssd"]["ssd2"]
        )
        image_storage_dir = (
            self.config["setting"]["ssd"]["ssd1"]
            if image_path == 0
            else self.config["setting"]["ssd"]["ssd2"]
        )

        # 更新配置
        self.config["setting"]["storage_params"]["video_path"] = os.path.join(
            video_storage_dir, "video", f"camera{self.config["setting"]["cameraid"]}"
        )
        self.config["setting"]["storage_params"]["image_path"] = os.path.join(
            image_storage_dir, "photo", f"camera{self.config["setting"]["cameraid"]}"
        )

        return True

    def set_detectmode(
        self,
        detect_mode: int,
        mask: int | None = None,
        tem_ltwh: List[int] | None = None,
    ) -> bool:
        # 0x31 0x32
        self.logger.debug(f"开始目标识别: {detect_mode}, {mask}, {tem_ltwh}")
        detector = None

        video_location = "/dev/null"
        image_location = "/dev/null"

        # 模式1和2共用VisionTracker逻辑
        if detect_mode in (1, 2):
            from .tracker import VisionTracker

            detector = VisionTracker(
                flip_mode=self.config["setting"]["encoder_params"]["flip_mode"], TPEs=3
            )
            if mask is not None:
                detector.set_target_mask(mask)
            detector.set_tracking_mode(detect_mode == 2)  # 模式2启用跟踪

        # 模式3处理模板跟踪
        elif detect_mode == 3:
            if not tem_ltwh:
                self.logger.warning("未指定模板, 自动退出模板匹配!")
                return False

            from .tracker import TemplateTracker

            detector = TemplateTracker(
                flip_mode=self.config["setting"]["encoder_params"]["flip_mode"],
            )

        else:
            self.logger.warning("不存在的目标检测方法!")

        if not detector:
            return False

        try:
            # 1. 通知预处理操作往目标检测队列放图
            # 2. 推理
            # 3. 关闭原始数据推流，开启目标检测结果推流
            start_args = (
                (detector, detect_mode, tem_ltwh)
                if detect_mode == 3
                else (detector, detect_mode)
            )

            if not self.imager_manager.assign_detworker(*start_args):
                self.logger.error("目标检测启动失败")
                return False

        except Exception as e:
            self.logger.error(f"检测流程异常: {str(e)}")
            return False

        if self.config["setting"]["storage_params"]["video_name"] == "/dev/null":

            video_location = self.config["setting"]["storage_params"]["video_name"]
            video_max_files = 0
        else:

            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            output_path_video = os.path.join(
                self.config["setting"]["storage_params"]["video_path"],
                "auto_save",
                "video",
                timestamp,
            )
            os.makedirs(output_path_video, exist_ok=True)

            dur = 10  # 默认分段时长为10秒

            # 计算能存储的文件数量
            parent_path = self.config["setting"]["storage_params"]["video_path"]
            mount_point = str(Path(parent_path).parents[1])
            video_max_files = self.compute_free_space(mount_point, "video", dur)
            if video_max_files <= 0:
                video_max_files = 0

            video_location = f"{output_path_video}/{self.config["setting"]["storage_params"]["video_name"]}"

        if self.config["setting"]["storage_params"]["image_name"] == "/dev/null":

            image_location = self.config["setting"]["storage_params"]["image_name"]
            image_max_files = 0
        else:

            output_path_image = os.path.join(
                self.config["setting"]["storage_params"]["image_path"],
                "auto_save",
                "image",
                timestamp,
            )
            os.makedirs(output_path_image, exist_ok=True)

            # 计算能存储的文件数量
            parent_path = self.config["setting"]["storage_params"]["image_path"]
            mount_point = str(Path(parent_path).parents[1])
            image_max_files = self.compute_free_space(mount_point, "image")

            if image_max_files <= 0:
                image_max_files = 0

            image_location = f"{output_path_image}/{self.config["setting"]["storage_params"]["image_name"]}"

        params = {
            "video_max_files": video_max_files,
            "video_location": video_location,
            "image_max_files": image_max_files,
            "image_location": image_location,
            "bind_port": self.config["setting"]["encoder_params"]["fake_bind_port"],
            "bps": self.config["setting"]["encoder_params"]["bps"],
            "udp_clients": f"""{self.config["setting"]["storage_params"]["video_ip"]}:{self.config["setting"]["storage_params"]["video_port"]},{self.config["setting"]["storage_params"]["image_ip"]}:{self.config["setting"]["storage_params"]["image_port"]}""",
        }

        if not self.imager_manager.assign_worker("live_encoder", **params):
            self.logger.error("重启原始推流失败!")
            return False

        params = {
            "bps": self.config["setting"]["encoder_params"]["bps"],
            "udp_clients": f"{self.config["setting"]["encoder_params"]["udp_ip"]}:{self.config["setting"]["encoder_params"]["udp_port"]}",
            "bind_port": self.config["setting"]["encoder_params"]["bind_port"],
        }

        if not self.imager_manager.assign_worker("live_det_encoder", **params):
            self.logger.error("目标检测结果推流失败!")
            return False

        return True

    def set_detectstop(self) -> bool:
        # 0x30
        self.logger.debug("停止目标识别")

        if "detect_activated" not in self.imager_manager.active_workers:
            self.logger.debug("该相机还没有开启目标检测，无需停止！")
            return True

        if not self.imager_manager.cancel_detworker():
            return False
        
        if not self.imager_manager.assign_worker("live_det_encoder"):
            self.logger.error("目标检测结果推流失败!")
            return False

        if not self.set_encode_setting(
            int(self.config["setting"]["encoder_params"]["bps"] // 1000),
            self.config["setting"]["encoder_params"]["fps"],
            scale=False,
        ):
            self.logger.error("重启原始推流失败!")
            return False

        return True

    async def save_capture_params(self):
        """
        Save self.config dict into the TOML file at cam_config_path,
        merging with any existing entries without wiping other sections.
        """
        # 1. 尝试读取已有的 TOML 文件
        cam_config_path = os.path.join(
            self.config["setting"]["cam_config_save_path"],
            self.config["setting"]["tittle"],
            self.config["setting"]["cam_config_path"],
        )

        try:
            async with aiofiles.open(cam_config_path, "r", encoding="utf-8") as f:
                content = await f.read()
                doc = parse(content)
        except (FileNotFoundError, IOError):
            # 文件不存在或无法读取，创建一个新的 TOML 表格
            doc = table()

        # 2. 合并新的配置
        for key, value in self.config.items():
            doc[key] = value

        # 3. 将合并后的内容写回文件（覆盖写入，但保留合并后的所有条目）
        async with aiofiles.open(cam_config_path, "w", encoding="utf-8") as f:
            await f.write(dumps(doc))

        return True

    def set_image_flip(self, mode: int) -> bool:
        """
        设置图像翻转
        :param mode: 0-正常 1-左右 2-上下 3-左右上下

        :return 执行结果
        """
        return True

    def set_cross_line(self, iscrossline_enable: int) -> bool:
        """
        设置图像中心十字线
        :param iscrossline_enable: 0-关闭 1-开启

        :return 执行结果
        """
        self.logger.debug(f"设置中心十字，参数：{iscrossline_enable}")
        if iscrossline_enable not in (0, 1):
            self.logger.error("无效的中心十字线参数")
            return False
        iscrossline_enable = True if iscrossline_enable == 1 else False
        if not self.imager_manager.update_preprocess_params(
            iscrossline_enable=iscrossline_enable
        ):
            return False
        return True

    def set_bpcorrection(self, bpcorrection_enable: int) -> bool:
        """
        设置坏点搜索
        :param bpcorrection_enable: 0-关闭 1-开启 2-盲元搜索 3-盲元保存

        :return 执行结果
        """

        if bpcorrection_enable not in (0, 1, 2, 3, 4):
            self.logger.error("无效的坏点校正参数")
            return False

        if bpcorrection_enable == 0:
            if not self.imager_manager.update_preprocess_params(bd_enable=False):
                self.logger.error("关闭坏点补偿失败")
                return False

        elif bpcorrection_enable == 1:
            # 开启坏点校正
            if not self.imager_manager.update_preprocess_params(bd_enable=True):
                self.logger.error("开启坏点补偿失败")
                return False

        elif bpcorrection_enable == 2:
            # 盲元搜索
            if not self.imager_manager.update_preprocess_params(
                bpcorrection_enable=True
            ):
                self.logger.error("盲元搜索失败")
                return False

        elif bpcorrection_enable == 3:
            # 盲元保存
            if not self.imager_manager.storage_blind_det_mask():
                self.logger.error("盲元保存失败")
                return False

        elif bpcorrection_enable == 4:
            # 关闭盲元搜索
            if not self.imager_manager.update_preprocess_params(
                bpcorrection_enable=False
            ):
                self.logger.error("关闭盲元搜索失败")
                return False

        self.logger.debug(f"设置坏点校正参数：{bpcorrection_enable}")
        return True

    def set_nuccorrection(self, nuc_enable: int) -> bool:
        """
        设置红外校正
        :param nuc_enable: 0-关闭 1-开启

        :return 执行结果
        """
        nuc_enable = True if nuc_enable == 1 else False

        if not self.imager_manager.update_preprocess_params(nuc_enable=nuc_enable):
            self.logger.error("更新红外校正参数失败")
            return False

        self.logger.debug(f"设置红外校正参数:{nuc_enable}")
        return True

    def set_autoenhance(self, autoenhance_enable: int) -> bool:
        """
        设置自动增强
        :param autoenhance_enable: 0-关闭 1-开启
        :return 执行结果
        """
        autoenhance_enable = True if autoenhance_enable == 1 else False
        if not self.imager_manager.update_preprocess_params(
            autoenhance_enable=autoenhance_enable
        ):
            self.logger.error("更新自动增强参数失败")
            return False
        self.logger.debug(f"设置自动增强参数:{autoenhance_enable}")
        return True
