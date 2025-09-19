import asyncio
import os
import tempfile
import logging
import threading
from typing import Awaitable, Callable, Dict, Any, Optional, List

try:
    import setproctitle

    HAS_SETPROCTITLE = True
except ImportError:
    HAS_SETPROCTITLE = False
    print("警告: setproctitle 未安装，进程标题设置将被跳过")

import zmq
import zmq.asyncio
import aioserial

from ..imager import ImageManagerWithPreInfra, ImageManagerWithPreVis
from ..ctrls import Cap2_Ctrl, Cap3_Ctrl, Cam2_Ctrl
from ..ctrls.v4l2ctrlor import v4l2_cmd
from ..bus.bus import bus
from ..utils import (
    parse_ip_port,
    get_bit_value,
    parse_bbox_from_uint32,
    map_boxes_to_crop_ltwh,
)
from ..config.config_manager import ConfigManager

idx_exposure_gain_map = {
    "vis_fix": 2,
    "vis_zoom": 2,
    "swir_fix": 2,
    "mwir_fix": 0,
    "lwir_fix": 0,
}


class UnifiedCameraController:
    """
    统一相机控制器
    支持多种相机模式：lwir_fix, mwir_fix, mwir_zoom, swir_fix, vis_fix, vis_zoom
    """

    def __init__(self, mode: str, config_manager: ConfigManager):
        """
        初始化统一相机控制器

        Args:
            mode: 相机模式 (lwir_fix, mwir_fix, mwir_zoom, swir_fix, vis_fix, vis_zoom)
            config_manager: 配置管理器
        """
        self.mode = mode
        self.config_manager = config_manager

        # 获取模式特定的配置
        self.cam_config = config_manager.get_camera_config(mode)
        self.imager_config = config_manager.get_imager_config(mode)

        self.camera_id = self.cam_config["setting"]["cameraid"]

        # 设置日志
        self.logger = logging.getLogger(f"unified_camera.{mode}")

        # 设置其他模块的日志记录器
        self._setup_module_loggers()

        # 根据模式选择控制器类型
        self._init_controller()

        # ZMQ上下文
        self.zmqContext = None

    def _setup_module_loggers(self):
        """设置其他模块的日志记录器"""
        try:
            # 设置v4l2ctrlor模块的日志记录器
            from ..ctrls.v4l2ctrlor import set_v4l2_logger

            set_v4l2_logger(self.logger)

            # 设置register_updater模块的日志记录器
            from ..setter.register_updater import set_register_updater_logger

            set_register_updater_logger(self.logger)

            self.logger.debug("模块日志记录器设置完成")
        except Exception as e:
            self.logger.warning(f"设置模块日志记录器失败: {e}")

    def _init_controller(self):
        """根据模式初始化控制器"""

        self.cam_ctrl = Cam2_Ctrl(
            cam_config=self.cam_config, bus=bus, logger=self.logger
        )

        # 根据模式选择控制器类型
        if self.mode in ["vis_fix", "vis_zoom"]:
            self.cap_ctrl = Cap3_Ctrl(
                cam_config=self.cam_config,
                imager_manager=self.imager_manager,
                cam_ctrl=self.cam_ctrl,
                logger=self.logger,
            )

        else:
            self.cap_ctrl = Cap2_Ctrl(
                cam_config=self.cam_config,
                imager_manager=self.imager_manager,
                logger=self.logger,
            )
        # 根据模式选择图像管理器类型
        if self.mode in ["mwir_zoom", "mwir_fix", "swir_fix", "lwir_fix"]:
            self.imager_manager = ImageManagerWithPreInfra(
                self.cam_config,
                self.imager_config,
                self.cam_config["setting"]["npz_path"],
                logger=self.logger,
            )
        else:
            self.imager_manager = ImageManagerWithPreVis(
                self.cam_config,
                self.imager_config,
                self.cam_config["setting"]["npz_path"],
                logger=self.logger,
            )

    async def parse_cap(self, data: dict) -> bool:
        """解析综控指令"""
        cmd = data.get("command", 0x00)
        args = data.get("parameters", [])

        # 特殊指令处理
        if cmd == 0x50:
            self.logger.debug("相机下电")
            t = threading.Thread(
                target=self.cap_ctrl.set_encode_switch, args=(0,), daemon=True
            )
            t.start()
            return True

        if cmd == 0x00:
            return True

        elif cmd == 0x11:
            # 网络接口设置
            address_ip, address_port = parse_ip_port(args[1], args[2])
            return self.cap_ctrl.set_addr(
                address_ip=address_ip, address_port=address_port
            )

        elif cmd == 0x29:
            # 实时视频开关
            encoder_switch = get_bit_value(args[0], self.camera_id)
            return self.cap_ctrl.set_encode_switch(encoder_switch=encoder_switch)

        elif cmd == 0x20:
            # 视频流编码设置
            return self.cap_ctrl.set_encode_setting(kbps=args[1], fps=args[2])

        elif cmd == 0x21:
            # 视频存储开关
            video_storage_switch = get_bit_value(args[0], self.camera_id)
            return self.cap_ctrl.set_video_storge(
                video_storage_switch=video_storage_switch
            )

        elif cmd == 0x22:
            # 视频存储设置
            return self.cap_ctrl.set_video_storge_setting(
                kbps=args[1], fps=args[2], dur=args[3]
            )

        elif cmd == 0x23:
            # 自动存图开关
            image_storage_switch = get_bit_value(args[0], self.camera_id)
            return self.cap_ctrl.set_image_storage(
                image_storage_switch=image_storage_switch,
                inter=args[2],
                max_files=args[1],
            )

        elif cmd == 0x24:
            # 立即存图开关
            return True

        elif cmd == 0x25:
            # 存储分配设置
            video_path = get_bit_value(args[0], self.camera_id)
            image_path = get_bit_value(args[1], self.camera_id)
            return self.cap_ctrl.set_storge_setting(
                video_path=video_path, image_path=image_path
            )

        elif cmd == 0x31:
            # 设置深度学习目标识别模型
            return self.cap_ctrl.set_detectmode(detect_mode=args[0], mask=args[2])

        elif cmd == 0x32:
            # 设置模板匹配识别
            bbox = parse_bbox_from_uint32(args[3], args[4])  # ==>[x_left, y_top, w, h]

            if self.mode in ["vis_zoom", "vis_fix"]:

                bbox = map_boxes_to_crop_ltwh(bbox)
                self.logger.debug(f"可见区域映射后坐标: {bbox}")

                if len(bbox) == 0:
                    self.logger.warning(f"可见模板不在区域范围内 无法进行模板匹配")
                    return True

            return self.cap_ctrl.set_detectmode(detect_mode=3, tem_ltwh=bbox)

        elif cmd == 0x30:
            # 停止目标识别
            return self.cap_ctrl.set_detectstop()

        elif cmd == 0xAE:
            # 保存参数到json文件，重启后加载
            return True
        else:
            self.logger.error("不存在的综控指令！")

        return False

    async def parse_cam(self, data: bytes) -> bool:
        cmd = data[1]
        args = data[2:]

        if cmd == 0x00:
            self.logger.debug("用户发送了空指令")
            return True

        elif cmd == 0x21:
            # return True
            hfov = args[1] * 0.01
            self.logger.debug(
                f"用户发送变焦连续调节指令: 方向：{args[0]} 期望水平变化视场角 {hfov} 度 "
            )
            return await self.cam_ctrl.v4l2_c_zoom(
                v4l2_cmd["CTRL_ZOOM_ABSOLUTE"],
                v4l2_cmd["CTRL_ZOOM_CURRENT"],
                args[0],
                hfov,
            )

        elif cmd == 0x22:
            hfov = args[0]
            self.logger.debug(f"用户发送变焦绝对位置指令: 水平视场角 {hfov} 度")

            ret, focallength = self.cam_ctrl._hfov2focallength(
                hfov
            )  # _hfov2focus会乘以0.1

            if not ret:
                self.logger.warning("计算焦距失败，直接返回咯！")
                return False

            vol = self.cam_ctrl._focus2vol(focallength)  # 写入电位器乘以4
            raw = int(vol * 4)
            raw = max(72, min(2704, raw))

            if hfov > 99:
                raw = 72
            elif hfov < 8:
                raw = 2704

            # self.logger.debug(f"计算得到焦距为: {focallength}, 电位器输出值为: {vol}*4={raw}")
            return await self.cam_ctrl.v4l2_w(
                v4l2_cmd["CTRL_ZOOM_ABSOLUTE"],
                raw,
            )

        elif cmd == 0x30:
            self.logger.debug(f"收到设置测控对对焦速度调节指令: {args[0]}")
            return await self.cam_ctrl.v4l2_w_and_r(
                v4l2_cmd["CTRL_FOCUS_SPEED"], args[0]
            )

        elif cmd == 0x31:
            self.logger.debug("用户发送对焦绝对位置指令")
            return await self.cam_ctrl.v4l2_w(v4l2_cmd["CTRL_FOCUS_ABSOLUTE"], args[0])

        elif cmd == 0x32:
            # return True
            self.logger.debug(
                f"用户发送对焦连续运动指令: 方向 {args[0]} 对焦量 {args[1]}"
            )  # args[0]=0/1
            return await self.cam_ctrl.v4l2_c_focus(
                v4l2_cmd["CTRL_FOCUS_ABSOLUTE"],
                v4l2_cmd["CTRL_FOCUS_CURRENT"],
                args[0],
                args[1],
            )

        elif cmd == 0x33:
            self.logger.debug("用户发送自动对焦指令-直接返回")
            return True

        elif cmd == 0x40:
            self.logger.debug("收到设置测控对连续光圈速度调节指令-直接返回")
            return True

        elif cmd == 0x41:
            self.logger.debug("收到设置测控对光圈调节指令")
            return True

        elif cmd == 0x50:

            idx = idx_exposure_gain_map.get(self.mode)
            if idx is None:
                self.logger.error("未知模式: %s，无法处理曝光调节指令", self.mode)
                return False

            try:
                raw_val = args[idx]
            except (IndexError, TypeError):
                self.logger.exception(
                    "曝光参数索引越界或参数类型错误: mode=%s, idx=%s, args=%s",
                    self.mode,
                    idx,
                    args,
                )
                return False

            try:
                val = int(raw_val)
            except (ValueError, TypeError):
                self.logger.exception(
                    "曝光参数无法转换为整数: %r (mode=%s)", raw_val, self.mode
                )
                return False

            self.logger.debug("用户发送曝光调节指令(%s): %s", self.mode, val)
            return await self.cam_ctrl.v4l2_w_and_r(v4l2_cmd["CTRL_EXPOSURE"], val)

        elif cmd == 0x51:
            idx = idx_exposure_gain_map.get(self.mode)
            if idx is None:
                self.logger.error("未知模式: %s，无法处理增益调节指令", self.mode)
                return False

            try:
                raw_val = args[idx]
            except (IndexError, TypeError):
                self.logger.exception(
                    "增益参数索引越界或参数类型错误: mode=%s, idx=%s, args=%s",
                    self.mode,
                    idx,
                    args,
                )
                return False

            try:
                val = int(raw_val)
            except (ValueError, TypeError):
                self.logger.exception(
                    "增益参数无法转换为整数: %r (mode=%s)", raw_val, self.mode
                )
                return False

            self.logger.debug("用户发送增益设置指令(%s): %s", self.mode, val)
            return await self.cam_ctrl.v4l2_w_and_r(v4l2_cmd["CTRL_ANALOGUE_GAIN"], val)

        elif cmd == 0x52:
            try:
                val = int(args[0] * 0.1)
            except (ValueError, TypeError):
                self.logger.exception("TEC温度参数无法转换为整数: %r", args[0])
                return False

            self.logger.debug(f"用户发送TEC温度设置指令:{args[0]} -> {val}")

            self.cam_ctrl.tec_temp = val  # 5度或者10度
            return await self.cam_ctrl.v4l2_w(
                v4l2_cmd["CTRL_TEMPERATURE_TEC"],
                val,
            )

        elif cmd == 0x64:
            self.logger.debug("用户发送盲元指令")
            return self.cap_ctrl.set_bpcorrection(bpcorrection_enable=args[0])

        elif cmd == 0x65:
            self.logger.debug("用户发送红外校正指令")
            return self.cap_ctrl.set_nuccorrection(nuc_enable=args[0])

        elif cmd == 0x71:
            self.logger.debug("用户发送图像增强指令")
            return self.cap_ctrl.set_autoenhance(autoenhance_enable=args[0])

        elif cmd == 0x80:
            self.logger.debug("用户发送图像翻转指令")
            return self.cap_ctrl.set_image_flip(mode=args[0])

        elif cmd == 0x81:
            self.logger.debug("用户发送十字线指令")
            return self.cap_ctrl.set_cross_line(iscrossline_enable=args[0])
        else:
            self.logger.debug("不存在的图像指令！")
            return False

    async def _control_service(
        self,
        *,
        name: str,  # 日志前缀：如 "综控" / "相机控制"
        entry_key: str,  # cam_config["setting"] 里的键：如 "cap_entry" / "cam_entry"
        handler: (
            Callable[[Any], Awaitable[Any]] | Callable[[Any], Any]
        ),  # parse_cap 或 parse_cam
        timeout: float = 2.0,
    ):
        """
        通用 ZMQ REP 服务：
        - 绑定到 cam_config["setting"][entry_key]
        - 收到请求后调用 handler(data) 并 send_pyobj(result)
        - 超时则继续轮询
        """
        sock = self.zmqContext.socket(zmq.REP)
        sock.setsockopt(zmq.LINGER, 0)
        # 可选：根据需求设置高水位，避免堆积
        # sock.setsockopt(zmq.RCVHWM, 100)
        # sock.setsockopt(zmq.SNDHWM, 100)

        tcp_path = self.cam_config["setting"][entry_key]
        sock.bind(f"tcp://{tcp_path}")
        self.logger.debug(
            f"{name}-{self.cam_config['setting']['cameraid']}: 绑定到 {tcp_path}"
        )

        try:
            while True:
                # 1) recv（带超时）
                try:
                    data = await asyncio.wait_for(sock.recv_pyobj(), timeout=timeout)
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    self.logger.exception("%s 接收出错: %s", name, e)
                    continue

                # 2) 处理
                try:
                    self.logger.debug(
                        f"{name}-{self.cam_config['setting']['cameraid']}: {data}"
                    )
                    if asyncio.iscoroutinefunction(handler):
                        result = await handler(data)
                    else:
                        # 如果是同步且可能 CPU 密集，放到线程池，防止阻塞事件循环
                        result = await asyncio.to_thread(handler, data)
                except Exception as e:
                    self.logger.exception("%s 处理出错: %s", name, e)
                    result = False

                # 3) 回复（REP 语义：每次 recv 之后必须 send 一次）
                try:
                    await sock.send_pyobj(result)
                except Exception as e:
                    self.logger.exception("%s 发送出错: %s", name, e)
                    # 发送失败通常是对端断开，继续循环等待下一次连接

        finally:
            # 任务取消或退出时，确保资源清理
            sock.close(0)

    async def start(self):
        """启动控制服务"""
        tasks = [
            self._control_service(
                self,
                name="综控",
                entry_key="cap_entry",
                handler=self.parse_cap,
                timeout=2.0,
            )
        ]
        if self.mode != "mwir_zoom":
            tasks.append(
                self._control_service(
                    self,
                    name="相机控制",
                    entry_key="cam_entry",
                    handler=self.parse_cam,
                    timeout=2.0,
                )
            )

        # 建议 return_exceptions=False，这样其中一个任务异常会直接抛出便于发现问题
        await asyncio.gather(*tasks, return_exceptions=False)

    def process(self):
        """主进程入口"""
        # 设置进程标题
        if HAS_SETPROCTITLE:
            setproctitle.setproctitle(self.mode)

        # 初始化ZMQ上下文
        self.zmqContext = zmq.asyncio.Context()

        # 启动异步事件循环
        asyncio.run(self.start())
