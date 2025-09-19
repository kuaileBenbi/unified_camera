from pathlib import Path
import queue
import threading
import traceback
import numpy as np

# from tracker.deepvisionTrack.YOlOv8.func import draw
from ..utils import draw
import cv2
from .nanoTracker import NnoTracker
from .core.config import cfg
import logging

# 默认日志记录器，会被具体的相机模式日志记录器替换
logger = logging.getLogger(__name__)


# 全局日志记录器设置函数
def set_templar_logger(camera_logger):
    """设置templar模块的日志记录器"""
    global logger
    logger = camera_logger


_CURRENT_DIR = Path(__file__).parent

config_path = _CURRENT_DIR / "models/config/config.yaml"
Tback_weight = _CURRENT_DIR / "models/track_backbone_T.rknn"
Xback_weight = _CURRENT_DIR / "models/track_backbone_X.rknn"
Head_weight = _CURRENT_DIR / "models/head.rknn"

cfg.merge_from_file(config_path)


class TemplateTracker:
    def __init__(self, flip_mode, callback=None):
        """
        Initialize the tracker with the provided frame, bounding box, and tracking method.

        Args:
            frame (ndarray): The initial frame in which the object is located.
            bbox (tuple): Bounding box in (x, y, w, h) format.
            method (str): The tracking method, either 'BOOSTING' or 'TM_CCOEFF_NORMED'.
        """

        self.flip_mode = flip_mode
        self.frame = None
        self.bbox = None
        self.template = None
        self.tracker = NnoTracker(cfg, Tback_weight, Xback_weight, Head_weight)

        self.callback = callback

        self._frame_queue = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()

        # Start worker thread
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

    def init_nano(self, frame, bbox):
        """
        bbox:xywh
        xy: 左上角坐标
        wh: 宽高
        """
        try:
            # cv2.imwrite("raw.jpg", frame)
            # roi = frame[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2], :]

            # cv2.imwrite("roi.jpg", roi)
            success = self.tracker.init(frame, bbox)
            if not success:
                logger.error("Tracker initialization failed")
                return False
        except Exception as e:
            logger.error(f"初始化遇到错误:{e}")
            return False
        return True

    def detworking(self, new_frame, frame_id=None):
        """
        Enqueue a new frame for tracking. Returns immediately.

        Args:
            new_frame (ndarray): the frame to process asynchronously
        """
        try:
            self._frame_queue.put_nowait((new_frame, frame_id))
        except queue.Full:
            try:
                self._frame_queue.get_nowait()
                self._frame_queue.put_nowait((new_frame, frame_id))
            except queue.Empty:
                pass

    def _process(self, frame):
        """
        Internal: run tracking on a single frame and prepare result.
        """
        try:
            outputs = self.tracker.track(frame)
            ltrb = list(map(int, outputs.get("bbox", [])))  # [l, t, r, b]

            if ltrb:
                # Draw bounding box and prepare result
                boxes = [ltrb]
                classes = ["tar"]
                result_frame = draw(
                    frame.copy(), boxes, classes, None, False, self.flip_mode
                )
                result = {"frame": result_frame, "boxes": boxes, "classes": classes}
                return result, True
            else:
                return {"frame": frame, "boxes": [], "classes": []}, False

        except Exception as e:
            logger.error(f"Tracking error: {e}")
            traceback.print_exc()
            return {"frame": frame, "boxes": [], "classes": []}, False

    def _worker(self):
        """
        Worker loop: fetch frames and process until stopped.
        """
        while not self._stop_event.is_set():
            try:
                frame, frame_id = self._frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            result, matched = self._process(frame)
            if matched:
                try:
                    self.callback(result, matched, frame_id)
                except Exception as cb_e:
                    logger.error(f"Callback error: {cb_e}")
            self._frame_queue.task_done()

    def stop_detworking(self):
        """
        Stop the worker thread and release tracker resources.
        """
        self._stop_event.set()
        self._worker_thread.join()

        try:
            released = self.tracker._release()
            if not released:
                logger.error("Failed to release tracker resources")
        except Exception as e:
            logger.error(f"Error releasing tracker: {e}")
