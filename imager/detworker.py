import threading
import queue
import time

import logging

import cv2

from .updater import DetResUpdater

# 默认日志记录器，会被具体的相机模式日志记录器替换
logger = logging.getLogger(__name__)


# 全局日志记录器设置函数
def set_detworker_logger(camera_logger):
    """设置detworker模块的日志记录器"""
    global logger
    logger = camera_logger


class DetectionWorker(DetResUpdater):
    def __init__(self, camera_tittle=None, logger=None):
        self.running = False
        self.detection_thread = None
        self.detector = None
        self.queues = None
        self.camera_tittle = camera_tittle

    @staticmethod
    def center_to_topleft(box: list[int]) -> list[float]:
        """
        将单个框由 [cx, cy, w, h] 转为 [x, y, w, h]
        其中 (x, y) 是左上角坐标。
        """
        cx, cy, w, h = box
        x = int(cx - w / 2)
        y = int(cy - h / 2)
        return [x, y, w, h]

    @staticmethod
    def remap_ltwh_to_crop(ltwh, crop_left=704, crop_top=704, crop_w=640, crop_h=640):
        """
        把原图 [l, t, w, h] 映射到裁剪图：
        - crop_left/crop_top: 裁剪在原图上的起始偏移
        - crop_w, crop_h: 裁剪后图像的宽高
        返回 [l', t', w', h']，并保证在 [0, crop_w]×[0, crop_h] 范围内。
        """
        l, t, w, h = ltwh

        # 1) 先减去偏移
        l_c = l - crop_left
        t_c = t - crop_top

        # 2) 边界裁剪 左上
        l_c = max(0, min(l_c, crop_w))
        t_c = max(0, min(t_c, crop_h))

        # 3) 宽高也要防出界
        w_c = max(0, min(w, crop_w - l_c))
        h_c = max(0, min(h, crop_h - t_c))

        return [int(l_c), int(t_c), int(w_c), int(h_c)]

    @staticmethod
    def crop_boxes_to_full_boxes(crop_boxes, full_size=(2048, 2048), crop_size=640):
        """
        将 crop 坐标系的框映射回原图坐标系 (left, top, right, bottom)。

        Args:
            crop_boxes: list of [x1, y1, x2, y2] in crop coords
            full_size: (H_full, W_full) of the原图
            crop_size: 边长 (假设正方形)

        Returns:
            orig_boxes: list of [x1, y1, x2, y2] in full coords
        """
        H_full, W_full = full_size
        dx = (W_full - crop_size) // 2
        dy = (H_full - crop_size) // 2

        orig_boxes = []
        for box in crop_boxes:
            x1, y1, x2, y2 = box
            orig_boxes.append(
                [
                    int(x1 + dx),
                    int(y1 + dy),
                    int(x2 + dx),
                    int(y2 + dy),
                ]
            )
        # print(f"cropbbox: {crop_boxes} -> origbbox: {orig_boxes}")
        return orig_boxes

    def start(self, detector, detection_mode, queues, *args) -> bool:

        if self.detection_thread is not None:
            if not self.stop():
                return False

        self.queues = queues

        self.running = True

        if detection_mode == 3:
            # 模板匹配
            self.set_mode(dl_model_mode=0, tem_model_mode=1)
        elif detection_mode == 2 or detection_mode == 1:
            # yolo+bytetrack
            # 只yolo
            self.set_mode(dl_model_mode=detection_mode, tem_model_mode=0)
        else:
            logger.warning(f"错误的跟踪模式: {detection_mode}")

        self.set_frame_mode(self.camera_tittle)

        target_func = (
            self._templar_worker if detection_mode == 3 else self._vision_worker
        )

        # 启动线程
        self.detection_thread = threading.Thread(
            target=target_func,
            args=(detector, args),
            daemon=True,
            name=f"Detectionthread-{detection_mode}",
        )
        self.detection_thread.start()

        return True

    def start_vis(self, detector, detection_mode, queues, *args) -> bool:

        if self.detection_thread is not None:
            if not self.stop():
                return False

        self.queues = queues

        self.running = True

        if detection_mode == 3:
            # 模板匹配
            self.set_mode(dl_model_mode=0, tem_model_mode=1)
        elif detection_mode == 2 or detection_mode == 1:
            # yolo+bytetrack
            # 只yolo
            self.set_mode(dl_model_mode=detection_mode, tem_model_mode=0)
        else:
            logger.warning(f"错误的跟踪模式: {detection_mode}")

        self.set_frame_mode(self.camera_tittle)

        target_func = (
            self._templar_worker_vis if detection_mode == 3 else self._vision_worker_vis
        )

        # 启动线程
        self.detection_thread = threading.Thread(
            target=target_func,
            args=(detector, args),
            daemon=True,
            name=f"Detectionthread-{detection_mode}",
        )
        self.detection_thread.start()

        return True

    def handle_result_deepvision(self, results, ret, frame_id=None):
        try:
            self.queues["enc_det"].put(results["frame"], timeout=0.1)
            # cv2.imwrite("det_enc.jpg", results["frame"])
            # print("放入队列成功")
        except queue.Full:
            # 丢 oldest，再插入
            try:
                _ = self.queues["enc_det"].get_nowait()
                self.queues["enc_det"].put_nowait(results["frame"])
            except Exception:
                pass

        self.update_status(results["classes"], results["boxes"])

    def handle_result_vis_deepvision(self, results, ret, frame_id=None):
        try:
            orig_boxes = self.crop_boxes_to_full_boxes(results["boxes"])
            self.queues["det_res"][frame_id] = (orig_boxes, results["classes"])
            # cv2.imwrite("det_enc.jpg", results["frame"])
            # print("放入队列成功")
        except Exception:
            logger.exception(f"可见目标检测报错")

        self.update_status(results["classes"], orig_boxes)

    def handle_result_template(self, results, ret, frame_id=None):
        try:
            self.queues["enc_det"].put(results["frame"], timeout=0.1)
            # cv2.imwrite("det_enc.jpg", results["frame"])
            # print("放入队列成功")
        except queue.Full:
            # 丢 oldest，再插入
            try:
                _ = self.queues["enc_det"].get_nowait()
                self.queues["enc_det"].put_nowait(results["frame"])
            except Exception:
                pass

        self.update_status(results["classes"], results["boxes"])

    def handle_result_vis_template(self, results, ret, frame_id=None):
        try:
            orig_boxes = self.crop_boxes_to_full_boxes(results["boxes"])
            self.queues["det_res"][frame_id] = (orig_boxes, results["classes"])
            # print("det_res_q", self.queues["det_res"])
            # cv2.imwrite("det_enc.jpg", results["frame"])
            # print("放入队列成功")
        except Exception:
            logger.exception(f"可见目标检测报错")

        self.update_status(results["classes"], orig_boxes)

    def _vision_worker_vis(self, detector, *args):
        """处理检测结果"""
        self.detector = detector
        detector.external_callback = self.handle_result_vis_deepvision

        while self.running:

            try:
                # 阻塞式取帧，0.1s 超时，拿到就处理，拿不到就继续 loop
                frame, pts = self.queues["det"].get(timeout=0.1)
                # cv2.imwrite("proc.jpg", frame)
                # 转换为网络能接受的rgb格式
                if len(frame.shape) < 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            except queue.Empty:
                # 没拿到就马上再试
                continue

            try:
                detector.detworking(frame, pts)
                # print(res.keys())
            except Exception:
                logger.exception("处理检测结果时发生异常")

    def _templar_worker_vis(self, detector, *args):
        """模板匹配工作线程"""
        self.detector = detector
        detector.callback = self.handle_result_vis_template
        temfirst = True
        template_bbox = args[0][0][0]
        ltwh_bbox = template_bbox

        logger.debug(f"设置模板匹配模板为ltwh: {template_bbox}")

        while self.running:
            # print("模板匹配")

            try:
                # 阻塞式取帧，0.1s 超时，拿到就处理，拿不到就继续 loop
                frame, pts = self.queues["det"].get(timeout=0.1)
                # cv2.imwrite("proc.jpg", frame)
                # 转换为网络能接受的BGR格式
                # frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                if frame.ndim < 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            except queue.Empty:
                continue

            if temfirst:

                if not detector.init_nano(frame, ltwh_bbox):
                    self.running = False
                    logger.debug("模板匹配初始化失败，已停止取图，请重启！")

                    if not detector.stop_detworking():
                        logger.debug("释放模板匹配rknn失败")
                    else:
                        logger.debug("成功释放模板匹配rknn，等待下一次重启！")

                temfirst = False
                continue

            detector.detworking(frame, pts)

    def _vision_worker(self, detector, *args):
        """处理检测结果"""
        self.detector = detector
        detector.external_callback = self.handle_result_deepvision

        while self.running:

            try:
                # 阻塞式取帧，0.1s 超时，拿到就处理，拿不到就继续 loop
                frame = self.queues["proc"].get(timeout=0.1)
                # cv2.imwrite("proc.jpg", frame)
                # 转换为网络能接受的rgb格式
                if len(frame.shape) < 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            except queue.Empty:
                # 没拿到就马上再试
                continue

            try:
                detector.detworking(frame)
                # print(res.keys())
            except Exception:
                logger.exception("处理检测结果时发生异常")

    def _templar_worker(self, detector, *args):
        """模板匹配工作线程"""
        self.detector = detector
        detector.callback = self.handle_result_template
        temfirst = True
        template_bbox = args[0][0][0]

        ltwh_bbox = template_bbox
        logger.debug(f"设置模板匹配模板为: ltrbwh >> {ltwh_bbox}")

        while self.running:
            # print("模板匹配")

            try:
                # 阻塞式取帧，0.1s 超时，拿到就处理，拿不到就继续 loop
                frame = self.queues["proc"].get(timeout=0.1)
                # cv2.imwrite("proc.jpg", frame)
                # 转换为网络能接受的BGR格式
                # frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            except queue.Empty:
                time.sleep(0.01)
                continue

            if temfirst:

                if not detector.init_nano(frame, ltwh_bbox):
                    self.running = False
                    logger.debug("模板匹配初始化失败，已停止取图，请重启！")

                    if not detector.stop_detworking():
                        logger.debug("释放模板匹配rknn失败")
                    else:
                        logger.debug("成功释放模板匹配rknn，等待下一次重启！")

                temfirst = False
                continue

            detector.detworking(frame)

    def stop_vis(self) -> bool:

        if not self.detection_thread or not self.detection_thread.is_alive():
            return True

        self.running = False

        try:
            if self.detection_thread and self.detection_thread.is_alive():
                self.detection_thread.join(timeout=1)
                self.detection_thread = None
        except Exception:
            logger.exception(f"停止目标检测线程时出错")
            return False

        if self.detector is not None and hasattr(self.detector, "stop_detworking"):
            self.detector.stop_detworking()  # 清理线程池

        self.detector = None

        queues = {"det": self.queues["det"], "det_res": self.queues["det_res"]}

        try:
            for n, q in queues.items():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

                del self.queues[n]

        except Exception:

            logger.exception(f"停止目标检测队列时出错")
            return False

        self.queues = None
        return True

    def stop(self) -> bool:

        if not self.detection_thread or not self.detection_thread.is_alive():
            return True

        self.running = False

        try:
            if self.detection_thread and self.detection_thread.is_alive():
                self.detection_thread.join(timeout=1)
                self.detection_thread = None
        except Exception:
            logger.exception(f"停止目标检测线程时出错")
            return False

        if self.detector is not None and hasattr(self.detector, "stop_detworking"):
            self.detector.stop_detworking()  # 清理线程池

        self.detector = None

        return True
