import os
import queue
import shlex
import traceback
import zlib
import cv2
import numpy as np
from string import Template
import time
import threading
import subprocess
import signal
from collections import deque

# import fcntl, os, time

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GLib", "2.0")
gi.require_version("cairo", "1.0")
from gi.repository import Gst, GLib, cairo

Gst.init(None)

import logging

# 默认日志记录器，会被具体的相机模式日志记录器替换
logger = logging.getLogger(__name__)

# 全局日志记录器设置函数
def set_gstworker_logger(camera_logger):
    """设置gstworker模块的日志记录器"""
    global logger
    logger = camera_logger


class CaptureDeadError(RuntimeError):
    """读图卡死异常"""

    pass


class GstPipeline:
    """GStreamer管道管理类"""

    def __init__(self, config, queues):
        self.config = config  # pipelines.yaml
        self.pipeline = None
        self.process = None
        self.queues = queues
        self.proc_running = False
        self.encoder_worker = None

        self.cap = None
        self.camera_wave = None
        self.params = None

        self._stop_event = threading.Event()
        self.lwir_stop_event = threading.Event()
        self.vis_stop_event = threading.Event()
        self.opencv_cap_t = None

        self.sample_queue = queue.Queue(maxsize=1)
        self.buffer_t = None
        self.buffer_t_vis = None

        self.v4l2_proc = None


    def preprocess_params(self, params: dict) -> dict:
        """参数预处理"""
        # 处理特殊参数类型
        if "segment_duration" in params:
            params["segment_duration"] = int(params["segment_duration"])

        # 处理混合格式参数
        params["auto_multicast"] = str(params.get("auto_multicast", "true")).lower()

        # 转义特殊字符
        return {k: self.escape_param(v) for k, v in params.items()}

    def escape_param(self, value) -> str:
        """参数转义处理"""
        if isinstance(value, bool):
            return str(value).lower()
        if isinstance(value, str) and any(c in value for c in (" ", "$", "{", "}")):
            return f'"{value}"'
        return str(value)

    def on_message(self, bus, message):
        """处理管道消息"""
        if message.type == Gst.MessageType.ERROR:
            err, _ = message.parse_error()
            logger.error(f"[{self.config["description"]}] 管道错误: {err.message}")
            self.stop()
        elif message.type == Gst.MessageType.EOS:
            logger.info(f"[{self.config["description"]}] 管道结束")
            self.stop()

    def on_new_sample_gray16(self, sink) -> Gst.FlowReturn:
        """原始视频帧处理回调"""
        # print("appsink 回调")
        try:
            sample = sink.emit("pull-sample")
            if not sample:
                logger.debug("not sample")
                return Gst.FlowReturn.ERROR
        except Exception as e:
            import traceback

            logger.error(f"处理帧失败: {e} \n {traceback.print_exc()}")
            return Gst.FlowReturn.ERROR

        try:
            # 快速往队列里塞 sample（只是个对象引用）
            self.sample_queue.put_nowait(sample)
        except queue.Full:
            # 如果队列满了，就丢最老的
            try:
                _ = self.sample_queue.get_nowait()
                self.sample_queue.put_nowait(sample)
            except queue.Empty:
                pass

        return Gst.FlowReturn.OK

    def _worker_buffer(self, img_type, dest_queue):

        while not self.lwir_stop_event.is_set():
            
            # 1) 阻塞等待 sample，周期性检查 stop
            try:
                sample = self.sample_queue.get(timeout=0.2)  # 200ms 超时，调大/调小均可
            except queue.Empty:
                continue  # 没有新样本，回到循环检查 STOP

            # 可选：用 None 作为“毒丸”让线程立刻退出
            if sample is None:
                break

            caps = sample.get_caps().get_structure(0)
            width  = caps.get_value("width")
            height = caps.get_value("height")

            buffer = sample.get_buffer()
            ok, info = buffer.map(Gst.MapFlags.READ)
            if not ok:
                continue

            try:
                # 避免额外拷贝：直接用 memoryview
                mv = memoryview(info.data)
                frame_gray = np.frombuffer(mv, dtype=np.uint16, count=width*height).reshape((height, width))
            finally:
                buffer.unmap(info)

            # 2) 投递到目标队列：短超时 + “满了就丢最旧”
            while not self.lwir_stop_event.is_set():
                try:
                    dest_queue.put(frame_gray, timeout=0.01)
                    break  # 成功放入，处理下一帧
                except queue.Full:
                    # 丢掉最旧的一帧以降低延迟
                    try:
                        _ = dest_queue.get_nowait()
                    except queue.Empty:
                        # 理论上很少发生；继续重试 put
                        pass
    
    def _worker_buffer_vis(self):

        while True:

            sample = self.sample_queue.get()

            caps = sample.get_caps().get_structure(0)
            width = caps.get_value("width")
            height = caps.get_value("height")

            buffer = sample.get_buffer()
            pts = buffer.pts
            ok, info = buffer.map(Gst.MapFlags.READ)

            if ok:
                raw = bytes(info.data)
                buffer.unmap(info)
                frame_bgr = np.frombuffer(raw, dtype=np.uint8).reshape(
                    (height, width, 3)
                )
                # cv2.imwrite("raw_vis.jpg", frame_bgr)

                try:
                    self.queues["det"].put((frame_bgr, pts), timeout=0.1)
                except queue.Full:
                    try:
                        _ = self.queues["det"].get(timeout=0.1)
                        self.queues["det"].put((frame_bgr, pts), timeout=0.1)

                    except queue.Empty:
                        pass

    def _start_v4l2_and_pipeline(self):
        cmd = f"""
                v4l2-ctl -d {self.params["device"]}
                    --set-fmt-video=width=1280,height=512,pixelformat=GREY
                    --stream-mmap=4
                    --set-selection=target=crop,flags=0,top=0,left=0,width=1280,height=512
                    --stream-to=-
                """
        logger.debug(f"LWIR_CMD: {cmd}")

        try:
            self.v4l2_proc = subprocess.Popen(
                shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            )
        except Exception as e:
            logger.error(f"开启硬件控制出错:{e},详细内容如下：")
            logger.error(traceback.format_exc())

            return False

        try:
            fd = self.v4l2_proc.stdout.fileno()
            pipeline_desc = f"""
                fdsrc fd={fd} !
                    videoparse format=gray16-le width=640 height=512 framerate=30/1 !
                    queue max-size-buffers=1 leaky=downstream !
                    appsink name=proc emit-signals=true sync=false max-buffers=1 drop=true
            """

            logger.debug(f"Generated pipeline string: {pipeline_desc}")
            self.pipeline = Gst.parse_launch(pipeline_desc)
            self.appsink = self.pipeline.get_by_name("proc")
            self.appsink.connect("new-sample", self.on_new_sample_gray16)
            self.pipeline.set_state(Gst.State.PLAYING)

        except Exception as e:
            logger.error(f"启动pipeline_desc发生错误: {e}")
            return False


    def read_y12_gray8(self, cap, width=640, height=512):
        if cap is None:
            return np.zeros((height, width), dtype=np.uint8)

        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to read frame from camera")
            return None

        img16 = frame.view(np.uint16).reshape(height, width)
        img8 = img16.astype(np.uint8)

        img8 = cv2.rotate(img8, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return img8

    def read_y12_gray12(self, cap, width=640, height=512):
        if cap is None:
            return np.zeros((height, width), dtype=np.uint16)
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to read frame from camera")
            return None

        img16 = frame.view(np.uint16).reshape(height, width)

        return img16

    def _worker_buffer_cv2cap(self, img_type, dest_queue):

        if img_type == "mwir_zoom":
            readtool = self.read_y12_gray8
        elif img_type == "fix":
            readtool = self.read_y12_gray12
        else:
            logger.debug(">>>> 图像类型错误！！！")
            return

        while not self._stop_event.is_set():
            try:
                frame = readtool(self.cap)
                if frame is None:
                    self._stop_event.wait(0.03)
                    continue
                try:
                    dest_queue.put(frame, timeout=0.01)
                    break  # 成功放入，处理下一帧
                except queue.Full:
                    # 丢掉最旧的一帧以降低延迟
                    try:
                        _ = dest_queue.get_nowait()
                    except queue.Empty:
                        # 理论上很少发生；继续重试 put
                        pass

            except Exception:
                logger.exception(f"读图卡死, 尝试重启")

    def _proc_loop(self, framerate, proc_name):
        """不断从 frame_queue 取出帧，做预处理后推给 appsrc"""
        count = 0
        interval = 1.0 / framerate

        if proc_name == "proc_encoder":
            logger.info(f"Encoder worker started for {self.config['description']}")
            try:
                enc_q = self.queues["enc"]
            except KeyError:
                logger.error("Encoder queue not found in queues")
                return

        elif proc_name == "proc_det_encoder":
            logger.info(
                f"Detection Encoder worker started for {self.config['description']}"
            )
            try:
                enc_q = self.queues["enc_det"]
            except KeyError:
                logger.error("Detection Encoder queue not found in queues")
                return

        logger.debug(f"推流取图线程开始了, self.proc_running = {self.proc_running}")

        while self.proc_running:

            try:
                frame_bgr = enc_q.get(timeout=0.1)
                # cv2.imwrite("enc.jpg", frame_bgr)
                # print(frame_bgr.shape, frame_bgr.mean())
            except queue.Empty:
                time.sleep(0.01)
                continue

            out_bytes = frame_bgr.tobytes()

            # 创建并填充 Gst.Buffer
            buf = Gst.Buffer.new_allocate(None, len(out_bytes), None)
            buf.fill(0, out_bytes)

            # 时间戳 & 时长（单位 ns）
            buf.pts = count * Gst.SECOND // framerate
            buf.duration = Gst.SECOND // framerate
            count += 1

            # 推给 appsrc
            ret = self.appsrc.emit("push-buffer", buf)
            if ret != Gst.FlowReturn.OK:
                logger.warning("appsrc push-buffer returned %s", ret)
                break

        logger.debug(
            f"编码取图线程 _proc_loop() {proc_name} 退出了，self.proc_running = {self.proc_running}"
        )

    def on_new_sample_vis(self, sink) -> Gst.FlowReturn:
        """原始视频帧处理回调"""
        try:
            sample = sink.emit("pull-sample")
            if not sample:
                logger.debug("not sample")
                return Gst.FlowReturn.ERROR
        except Exception as e:
            import traceback

            logger.error(f"处理帧失败: {e} \n {traceback.print_exc()}")
            return Gst.FlowReturn.ERROR

        try:
            # 快速往队列里塞 sample（只是个对象引用）
            self.sample_queue.put_nowait(sample)
        except queue.Full:
            # 如果队列满了，就丢最老的
            try:
                _ = self.sample_queue.get_nowait()
                self.sample_queue.put_nowait(sample)
            except queue.Empty:
                pass

        return Gst.FlowReturn.OK

    def draw_cb(self, ovl, context, timestamp, duration):
        """
        cairooverlay draw 回调：timestamp 就是 PTS（ns），
        用它去取检测分支存下的框，保证对应。
        """

        if self.box_map is None or len(self.box_map.keys()) < 1:
            return

        boxes, classes = self.box_map.pop(timestamp, ([], []))  # (boxes:[], classes:[])

        if len(boxes) < 1:
            return

        context.set_line_width(2)
        context.set_source_rgb(1, 0, 0)  # 红框
        # 设置字体
        context.select_font_face("Sans")
        context.set_font_size(24)  # 字号，可根据需要调整

        # print(f"Box @PTS {timestamp}: {boxes}")
        for _, (box, cla) in enumerate(zip(boxes, classes)):
            x1, y1, x2, y2 = box
            w = x2 - x1  # 计算宽度
            h = y2 - y1  # 计算高度

            x0 = x1 + self.left  # 左上角 x 坐标加上偏移量
            y0 = y1 + self.top  # 左上角 y 坐标加上偏移量

            context.rectangle(x0, y0, w, h)

            text = f"ID:{cla}"
            context.move_to(x0 + 5, y0 - 5)
            context.show_text(text)
            context.stroke()

    def _build_pipeline(self, params: dict) -> bool:
        """构建GStreamer管道"""
        # 参数预处理
        params = self.preprocess_params(params)
        # 生成管道字符串
        pipeline_str = Template(self.config["template"]).safe_substitute(params)
        logger.debug(f"Generated pipeline string: {pipeline_str}")
        try:
            self.pipeline = Gst.parse_launch(pipeline_str)
        except Exception as e:

            logger.error(f"生成pipeline出错: {e}")
            return False

        if "proc_encoder" in pipeline_str:

            self.frame_duration = Gst.SECOND // int(params["raw_fps"])

            try:
                self.appsrc = self.pipeline.get_by_name("proc_encoder")
                self.proc_running = True
                self.encoder_worker = threading.Thread(
                    target=self._proc_loop,
                    args=(int(params["raw_fps"]), "proc_encoder"),
                    daemon=True,
                )
                self.encoder_worker.start()

            except Exception as e:
                logger.error(f"设置appsrc回调函数出错: {e}")
                return False

        if "proc_det_encoder" in pipeline_str:

            self.frame_duration = Gst.SECOND // int(params["raw_fps"])

            try:
                self.appsrc = self.pipeline.get_by_name("proc_det_encoder")
                self.proc_running = True
                self.encoder_worker = threading.Thread(
                    target=self._proc_loop,
                    args=(int(params["raw_fps"]), "proc_det_encoder"),
                    daemon=True,
                )
                self.encoder_worker.start()

            except Exception as e:
                logger.error(f"设置appsrc回调函数出错: {e}")
                return False
        
        if "ovl" in pipeline_str:
            self.ovl = self.pipeline.get_by_name("ovl")
            self.ovl.connect("draw", self.draw_cb)
            self.box_map = self.queues["det_res"]
            self.top = int(params["top"])
            self.bottom = int(params["bottom"])
            self.left = int(params["left"])
            self.right = int(params["right"])
            self.crop_width = int(params["crop_width"])
            self.crop_height = int(params["crop_height"])

        if "vis_det" in pipeline_str:
            self.buffer_t_vis = threading.Thread(target=self._worker_buffer_vis, daemon=True)
            self.buffer_t_vis.start()
            self.appsink = self.pipeline.get_by_name("vis_det")
            self.appsink.connect("new-sample", self.on_new_sample_vis)

        # 总线消息处理
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.on_message)

        return True

    def stop(self) -> bool:
        """停止流水线"""
        if self.buffer_t_vis and self.buffer_t_vis.is_alive():
            self.vis_stop_event.set()
            try:
                self.buffer_t_vis.join(timeout=1)
                self.buffer_t_vis = None
            except Exception as e:
                logger.error(f"停止worker_buffer_vis发生错误:{e}")
                return False
        
        if self.buffer_t and self.buffer_t.is_alive():
            self.lwir_stop_event.set()
            try:
                self.buffer_t.join(timeout=1)
                self.buffer_t = None
            except Exception as e:
                logger.error(f"停止worker_buffer发生错误:{e}")
                return False
        
        if self.opencv_cap_t and self.opencv_cap_t.is_alive():
            self._stop_event.set()
            try:
                self.opencv_cap_t.join(timeout=1)
                self.opencv_cap_t = None
            except Exception as e:
                logger.error(f"停止worker_buffer_cv2cap发生错误:{e}")
                return False

        if self.pipeline is not None:
            try:
                self.pipeline.set_state(Gst.State.PAUSED)
                self.pipeline.set_state(Gst.State.NULL)
                self.pipeline.get_state(timeout=3 * Gst.SECOND)
                logger.info(f"stop pipeline: {self.config["description"]}")
                self.pipeline = None
            except Exception as e:
                logger.error(f"设置pipeline状为NULL发生错误: {e}")
                return False

        if self.process is not None:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(5)
                self.process = None
                logger.info(f"kill pipeline: {self.config["description"]}")
            except Exception as e:
                logger.error(f"kill当前pipeline发生错误: {e}")
                return False

        if self.cap is not None:
            try:
                self.cap.release()
                self.cap = None
                logger.info(f"释放摄像头资源: {self.config["description"]}")
            except Exception as e:
                logger.error(f"释放摄像头资源发生错误: {e}")
                return False

        if self.v4l2_proc is not None:
            try:
                self.lwir_stop_event.set()
                time.sleep(0.05)
                self.v4l2_proc.send_signal(signal.SIGINT)
                self.v4l2_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.v4l2_proc.kill()
                self.v4l2_proc.wait(timeout=2)
            except Exception as e:
                logger.error(f"kill当前pipeline发生错误: {e}")
                return False
            finally:
                self.v4l2_proc = None

        return True

    def _start_capture(self):
        """真正打开相机、启动线程"""
        if self.cap is not None:
            self.cap.release()
            time.sleep(0.05)

        if self.camera_wave is None or self.params is None:
            logger.error("摄像头波段或参数未设置，无法启动捕获")
            return False

        self.cap = cv2.VideoCapture(self.params["device"])
        if not self.cap.isOpened():
            logger.error(f"无法打开设备: {self.params['device']}")
            return False

        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # 丢弃残留帧
        for _ in range(2):
            self.cap.grab()

        self.opencv_cap_t = threading.Thread(
            target=self._worker_buffer_cv2cap,
            args=(self.camera_wave, self.queues["raw"]),
            daemon=True,
        )
        self.opencv_cap_t.start()
        return True

    def start(self, params: dict) -> bool:
        """启动流水线"""

        if not self.stop():
            time.sleep(0.05)
            return False

        if self.config["type"] == "gst":
            if not self._build_pipeline(params):
                return False
            self.pipeline.set_state(Gst.State.PLAYING)

        elif self.config["type"] == "subprocess":
            cmd = ["gst-launch-1.0"] + Template(
                self.config["template"]
            ).safe_substitute(params).split()
            logger.debug(f"cmd: {cmd}")
            try:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid,
                )
            except Exception as e:
                logger.error(f"启动Popen发生错误: {e}")
                return False

        elif self.config["type"] == "v4l2_mwir_zoom":
            self.params = params
            self.camera_wave = params.get("camera_wave", "mwir_zoom")
            if not self._start_capture():
                logger.warning("打开V4L2摄像头失败！")
                return False

        elif self.config["type"] == "v4l2_mwir_swri_fix":
            self.params = params
            self.camera_wave = params.get("camera_wave", "fix")
            if not self._start_capture():
                logger.warning("打开V4L2摄像头失败！")
                return False

        elif self.config["type"] == "v4l2_lwir_fix":

            self.params = params

            self.buffer_t = threading.Thread(
                target=self._worker_buffer,
                args=("fix", self.queues["raw"]),
                daemon=True,
            )
            self.buffer_t.start()

            try:
                self._start_v4l2_and_pipeline()
            except Exception as e:
                logger.error(f"开启硬件控制出错:{e},详细内容如下：")
                logger.error(traceback.format_exc())
                return False

        else:
            logger.error("不存在的gst执行类型! ")
            return False

        logger.info(f"已启动pipeline: {self.config["description"]}")
        return True
