from collections import deque
import logging

# 默认日志记录器，会被具体的相机模式日志记录器替换
logger = logging.getLogger(__name__)


# 全局日志记录器设置函数
def set_preworker_logger(camera_logger):
    """设置preworker模块的日志记录器"""
    global logger
    logger = camera_logger


import threading
import time
import queue
import math
import cv2
import numpy as np
from typing import Optional, Tuple
import os
from datetime import datetime

from status import set_values

try:
    from .preutils import ImagePreprocessor  # 相对导入
except ImportError:
    from preutils import ImagePreprocessor  # 直接导入（用于脚本直接运行


class PreprocessWorker:
    def __init__(self, camconfig: dict = None, npz_path: str = None):

        self.cur_wave = camconfig["setting"]["wave"]
        self.cur_identity = camconfig["setting"]["tittle"]
        self.max_val = 16383 if self.cur_identity == "lwir" else 4095
        self.detlock = threading.Lock()
        self.paramlock = threading.Lock()
        self.isdet = False

        self.storage_queue = deque(maxlen=1)
        self.clarity_queue = deque(maxlen=1)

        self.running = False
        self.imgprecessor = ImagePreprocessor(npz_path)

        # 初始化所有预处理操作为false
        self.init_preprocess_params()

        self.saving = False
        self.stop_event = threading.Event()
        self.save_thread = None
        storage_base_dir = camconfig["setting"]["storage_params"]["image_path"]
        self.storage_dir = os.path.join(storage_base_dir, "raw")
        os.makedirs(self.storage_dir, exist_ok=True)

    def _load_calibration(self, temp_int: int):
        self.imgprecessor.load_correction_params(temp_int=temp_int)

    def init_preprocess_params(self):

        self.flip_mode = 0

        self.autoenhance_enable = False

        self.bpcorrection_enable = False

        self.blind_mask_det = None

        self.iscrossline_enable = False
        self.isflip_enable = False
        self.nuc_enable = True
        self.bd_enable = True

        self.allowed_params = {
            "autoenhance_enable",
            "bpcorrection_enable",
            "iscrossline_enable",
            "isflip_enable",
            "flip_mode",
            "nuc_enable",
            "bd_enable",
        }

    def update_preprocess_params(self, **kwargs):
        """
        更新预处理开关参数。

        返回:
            updated_keys (list of str): 成功更新的参数名列表。
            invalid_keys (list of str): 无效的参数名列表（未被更新）。
        """
        updated_keys = []
        invalid_keys = []
        with self.paramlock:
            for key, value in kwargs.items():
                if key in self.allowed_params:
                    setattr(self, key, value)
                    updated_keys.append(key)
                else:
                    logger.error(
                        f"Invalid parameter '{key}' for update_preprocess_params"
                    )
                    invalid_keys.append(key)
        return updated_keys, invalid_keys

    def update_blind_mask(self, new_blink_mask):
        return np.logical_or(self.blind_mask_det, new_blink_mask).astype(np.uint8)

    def storage_blind_det_mask(self):
        if not self.imgprecessor.storge_blindpixels(self.blind_mask_det):
            logger.error("存储坏点检测掩码失败，请检查路径和权限")
            return False
        return True

    def set_detflag(self, value):

        with self.detlock:
            self.isdet = value

    def async_blind_detection(self, frame):
        try:
            new_mask = self.imgprecessor.apply_blind_pixel_detect(frame)
            self.blind_mask_det = new_mask
        except Exception as e:
            logger.debug("盲元检测错误:", e)

    def preprocess_ctrl(self, frame):

        if self.cur_identity == "mwir_zoom":
            return frame

        frame = self.imgprecessor.default_process(frame, self.cur_identity)  # 16位图像

        # 坏点检测
        if self.bpcorrection_enable:
            # print("bpcorrection_enable")
            if self.blind_mask_det is None:
                logger.debug("首次坏点检测")
                detection_frame = frame.copy()
                threading.Thread(
                    target=self.async_blind_detection,
                    args=(detection_frame,),
                    daemon=True,
                ).start()

            else:
                logger.debug("应用坏点补偿")
                frame = self.imgprecessor.compensate_blind_pixels(
                    frame, self.blind_mask_det
                )

        # 自动图像增强
        if self.autoenhance_enable:
            frame = self.imgprecessor.apply_autogian(frame)

        # 中心十字显示
        if self.iscrossline_enable:
            frame = self.imgprecessor.draw_center_cross_polylines(frame)

        frame_8bit = self.imgprecessor.scale_to_8bit(frame, self.max_val)

        frame_8bit = cv2.flip(frame_8bit, 0)  # 预处理后翻转

        return frame_8bit

    @staticmethod
    def tenengrad_calculator(
        img: np.ndarray,
        sobel_ksize: int = 3,
        downsample_factor: float = 0.5,
        grad_threshold: float = 0.0,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> float:
        """
        计算图像的 Tenengrad 清晰度评价值。

        参数
        ----
        img : np.ndarray
            原始图像，支持灰度或彩色（自动转灰度）。
        sobel_ksize : int
            Sobel 卷积核大小，常用 3 或 5。
        downsample_factor : float
            下采样比例 (0,1]，如 0.5 表示缩至原来一半，以加速计算。
        grad_threshold : float
            梯度能量阈值，小于该值的像素点在累加前被置零，用于剔除噪声。
        roi : (x, y, w, h) 可选
            区域兴趣，四元组指定裁剪区域后再做下采样和梯度计算。
            若为 None，则全图计算。

        返回
        ----
        float
            Tenengrad 焦点度量值，越大代表越清晰。
        """
        # 1. 裁剪 ROI
        if roi is not None:
            x, y, w, h = roi
            img = img[y : y + h, x : x + w]

        # 2. 下采样加速
        if downsample_factor != 1.0:
            new_w = int(img.shape[1] * downsample_factor)
            new_h = int(img.shape[0] * downsample_factor)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 3. 转为灰度
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # 4. 计算 Sobel 梯度
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=sobel_ksize)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=sobel_ksize)

        # 5. 梯度能量并加阈去噪
        grad2 = gx * gx + gy * gy
        if grad_threshold > 0:
            grad2 = np.where(grad2 >= grad_threshold, grad2, 0.0)

        # 6. 累加并返回
        return float(np.sum(grad2))

    def measure_sharpness(self, fm_kwargs):
        """
        测量图像的清晰度
        """
        try:
            frame = self.clarity_queue.pop()
        except IndexError:
            return

        sharpness = self.tenengrad_calculator(img=frame, **fm_kwargs)
        scaled16 = int(65535 * math.log1p(sharpness) / math.log1p(1e9))
        set_values({f"{self.cur_identity}_sharpness_evaluation": int(scaled16)})

    def _schedule_loop(self, scheduler, fm_kwargs, interval):
        """在独立线程里运行，非阻塞主逻辑"""
        # print("计算清晰度")
        scheduler.enter(
            interval, 1, self._schedule_loop, (scheduler, fm_kwargs, interval)
        )
        self.measure_sharpness(fm_kwargs)

    def save_images(self, interval: float, num_frames: int, prefix="frame"):
        """
        新开线程保存图像
        interval: 保存间隔秒
        num_frames: 保存张数
        prefix: 保存文件名前缀
        """
        if interval < 0 or num_frames <= 0:
            logger.error("保存参数错误，间隔应>=0，张数应>0")
            return False

        # 如果已有保存任务 -> 请求停止
        if self.saving and self.save_thread is not None and self.save_thread.is_alive():
            logger.info("检测到已有保存任务，准备停止并重启...")
            self.stop_event.set()
            self.save_thread.join()  # 等待线程退出
            self.stop_event.clear()

        # print(f"现在开始 按照间隔 {interval} 秒 保存{num_frames} 张图像！")

        def _worker():
            self.saving = True
            count = 0

            try:
                while count < num_frames and not self.stop_event.is_set():

                    try:
                        frame, processed_frame = self.storage_queue.popleft()
                    except IndexError:
                        time.sleep(0.03)
                        continue

                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

                    cur_it_ms = self.imgprecessor.cur_t_ms
                    cur_temp_c_t = self.imgprecessor.cur_temp_c_t

                    frame_filename = os.path.join(
                        self.storage_dir,
                        f"{prefix}_{cur_temp_c_t}_{cur_it_ms}_{ts}_{count:04d}.png",
                    )
                    processed_filename = os.path.join(
                        self.storage_dir,
                        f"processed_{cur_temp_c_t}_{cur_it_ms}_{ts}_{count:04d}.png",
                    )

                    cv2.imwrite(frame_filename, frame)
                    cv2.imwrite(processed_filename, processed_frame)

                    logger.info(
                        "保存图像 %s 和处理图像 %s 成功",
                        frame_filename,
                        processed_filename,
                    )

                    count += 1
                    time.sleep(interval)

            finally:
                self.storage_queue.clear()
                self.saving = False  # <--- 保存结束，释放标志
                logger.info("图像保存线程退出，已清空队列。")

        self.save_thread = threading.Thread(target=_worker, daemon=True)
        self.save_thread.start()

        return True

    @staticmethod
    def _put_drop_oldest(q: queue.Queue, item) -> bool:
        """仅用于 enc/proc 这样的 queue.Queue：满则丢最旧再放入。"""
        try:
            q.put_nowait(item)
            return True
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(item)
                return True
            except queue.Full:
                return False

    def run(self, queues):
        """预处理流水线"""
        logger.info("预处理线程启动")

        q_raw = queues["raw"]  # queue.Queue
        q_enc = queues["enc"]  # queue.Queue
        q_proc = queues["proc"]  # queue.Queue

        while self.running:

            try:
                frame = q_raw.get(timeout=0.03)  # 不翻转
            except queue.Empty:
                continue

            try:
                processed = self.preprocess_ctrl(frame)  # 校正之后翻转
            except Exception as e:
                logger.exception("预处理失败，丢弃此帧：%s", e)
                continue

            self.storage_queue.append(
                (frame, processed.copy())
            )  # 不翻转原始图和反转后的预处理图
            self.clarity_queue.append(processed)

            self._put_drop_oldest(q_enc, processed)
            self._put_drop_oldest(q_proc, processed)

        logger.warning("预处理线程 run() 退出了，self.running = %s", self.running)

    def stop(self):
        self.running = False
