import asyncio
import logging
import math
import queue
import time

import cv2
import numpy as np

from lwir_fix.setter import update_registers
from .v4l2ctrlor import (
    v4l2_cmd,
    set_camera_param,
    get_camera_param,
    pack_temp_integration,
    exposure_tool_n2t,
    exposure_tool_t2n,
)
from .calibrator import pot_to_focallength, focallength_to_pot
from status import get_values
from ..utils.utils import hex_to_decimal_with_fraction

# 默认日志记录器，会被具体的相机模式日志记录器替换
logger = logging.getLogger(__name__)


default_bounds = {
    "swir_fix": (12, 25718),
    "mwir_fix": (12, 26324),
    "lwir_fix": (12, 26325),
    "vis_fix": (733, 26379),
    "vis_zoom": (204, 2200),
}


class CamCtrl:

    def __init__(self, cam_config, bus=None, logger=None):

        self.camera_addr = cam_config["setting"]["video_devices_ctrl"]
        self.sensor_width_mm = cam_config["setting"]["sensor"]["size_mm"]
        self.camera_identity = cam_config["setting"]["tittle"]
        self.wave = cam_config["setting"]["wave"]
        self.tec_temp = 5  # 默认TEC温度
        self.bus = bus

        # 设置日志记录器
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)

        # 设置tracker包中其他模块的日志记录器
        self._setup_tracker_module_loggers()

    def _setup_tracker_module_loggers(self):
        """设置tracker包中其他模块的日志记录器"""
        try:
            # 设置templar模块的日志记录器
            from .tracker.templaterTrack.templar import set_templar_logger

            set_templar_logger(self.logger)

            # 设置visionar模块的日志记录器
            from .tracker.deepvisionTrack.visionar import set_visionar_logger

            set_visionar_logger(self.logger)

            self.logger.debug("tracker包模块日志记录器设置完成")
        except Exception as e:
            self.logger.warning(f"设置tracker包模块日志记录器失败: {e}")

    def set_integration_time(self, temp_int: int):
        self.bus.emit("integration.change", temp_int)
        self.logger.debug("[CamCtrl] 已发布 integration.change")

    def _focallength2hfov(self, focal_length_mm: int) -> float:
        """
        由焦距计算水平视场角 HFOV。
        :param focal_length: 焦距值 单位为mm
        :return: HFOV 单位为度
        """
        # print(
        #     f"焦距为: {focal_length_mm} mm, sensor_width_mm: {self.sensor_width_mm}mm"
        # )
        hfov_rad = 2 * math.atan(
            ((self.sensor_width_mm * 2048) / 2) / (focal_length_mm + 1e-6)
        )
        hfov_deg = math.degrees(hfov_rad)
        return hfov_deg

    def _hfov2focallength(self, hfov: int) -> int:
        """
        由水平视场角计算焦距
        :param hfov: 水平视场角 uint32 lsb=0.1度
        :return: 焦距mm/0.1
        """
        hfov *= 0.1
        self.logger.debug(f"目的水平视场角：{hfov}")
        # 焦距计算公式: f = (sensor_width) / (2 * tan(FOV / 2))
        try:
            focal_length = (self.sensor_width_mm * 2048) / (
                2 * math.tan(math.radians(hfov / 2))
            )
        except ZeroDivisionError:
            self.logger.warning("视场角导致除以零错误。请检查输入参数。")
            return False, 0

        return True, focal_length

    @staticmethod
    def _vol2focallength(vol: float) -> float:
        """
        由电位器输出值计算焦距值
        """
        return pot_to_focallength(vol)

    @staticmethod
    def _focallength2vol(focallength: float) -> float:
        """
        由焦距计算出电位器输出值
        """
        return focallength_to_pot(focallength)

    async def v4l2_c_zoom(
        self,
        w_param_id: int,
        r_param_id: int,
        value: int,
        hfov: int,
    ) -> bool:
        """
        发送v4l2只写变焦连续运动指令 直到运行到给定差量
        :param_id: 控制指令
        :value: 控制方向 0:增加 1:减少
        :hfov: 调节量
        :return: 如果设置成功返回 True 否则返回 False
        """
        try:
            # 获取相机当前视场角
            cur_vol = await asyncio.to_thread(
                get_camera_param, self.camera_addr, r_param_id
            )
        except Exception:
            self.logger.exception(f"设置参数 {w_param_id} 时发生错误")
            return False

        await asyncio.sleep(0.05)

        if cur_vol < 72 or cur_vol > 2708:
            self.logger.warning(f"驱动器返回不在正常值，无法进行变焦调节！")
            return False

        cur_focus = self._vol2focallength(cur_vol // 4)
        cur_hfov = self._focallength2hfov(cur_focus)
        update_registers({"hfov": int(round(cur_hfov * 10))}, self.camera_identity)

        # print(f"cur_vol:{cur_vol}, cur_focus: {cur_focus}, cur_fov:{cur_hfov}")

        hope_fov = cur_hfov + hfov if value == 0 else cur_hfov - hfov

        if hope_fov > 9.92:
            return await self.v4l2_w(
                w_param_id,
                r_param_id,
                72,
            )

        elif hope_fov < 0.87:
            return await self.v4l2_w(
                w_param_id,
                r_param_id,
                2704,
            )

        ret, hope_focus = self._hfov2focallength(hope_fov * 10)  # _hfov2focus的lsb=0.1

        if not ret:
            self.logger.warning("计算焦距失败，直接返回咯！")
            return False

        hope_vol = self._focus2vol(hope_focus)

        # print(
        #     f"hope_fov: {hope_fov}, hope_focus: {hope_focus}, hope_vol:{hope_vol}*4={hope_vol*4}"
        # )

        return await self.v4l2_w(
            w_param_id,
            r_param_id,
            int(hope_vol * 4),
        )

    async def v4l2_c_focus(
        self,
        w_param_id: int,
        r_param_id: int,
        value: int,
        focus_jog: int,
    ) -> bool:
        """
        发送v4l2只写对焦连续运动指令 直到运行到给定差量
        :param_id: 控制指令
        :value: 控制方向 0:增加 1:减少
        :focus_jog: 调节量
        :return: 如果设置成功返回 True 否则返回 False
        """

        cam_id = getattr(self, "camera_identity", None)
        min_focus, max_focus = default_bounds.get(cam_id, (12, 26325))

        # —— 读取当前对焦位置 ——
        try:
            cur_focus = await asyncio.to_thread(
                get_camera_param, self.camera_addr, r_param_id
            )
        except Exception:
            self.logger.exception(
                "读取当前对焦位置失败: addr=%s, rid=%s",
                getattr(self, "camera_addr", None),
                r_param_id,
            )
            return False

        await asyncio.sleep(0.05)  # 保留原有轻微延时

        # —— 合法性检查 ——
        if not (min_focus <= int(cur_focus) <= max_focus):
            self.logger.warning(
                "当前对焦值越界, cam=%s, cur=%s, expect=[%s,%s], 取消调节",
                cam_id,
                cur_focus,
                min_focus,
                max_focus,
            )
            return False

        # —— 状态回写（当前值）——
        try:
            update_registers({"focus_abs_position": int(cur_focus)}, cam_id)
        except Exception:
            self.logger.exception(
                "回写当前对焦位失败: cam=%s, cur=%s", cam_id, cur_focus
            )
            return False

        # —— 计算目标对焦位置（相对移动 & 方向约定：0=增加；1=减少）——
        direction = +1 if value == 0 else -1
        step = abs(int(focus_jog))
        hope_focus = int(cur_focus) + direction * step

        # —— 边界裁剪 ——
        if hope_focus < min_focus:
            hope_focus = min_focus
        elif hope_focus > max_focus:
            hope_focus = max_focus

        # self.logger.debug("对焦: cur=%s, step=%s(%s), target=%s, bounds=[%s,%s]",
        #                   cur_focus, step, "inc" if direction>0 else "dec", hope_focus, min_focus, max_focus)

        # —— 下发目标位置 ——
        try:
            return await self.v4l2_w(w_param_id, r_param_id, hope_focus)
        except Exception:
            self.logger.exception(
                "写入目标对焦失败: wid=%s, rid=%s, target=%s, cam=%s",
                w_param_id,
                r_param_id,
                hope_focus,
                cam_id,
            )
            return False

    async def v4l2_w(self, w_param_id: int, value: int) -> bool:
        """
        发送v4l2只写指令: 镜头: 绝对对焦/变焦指令; TEC温度
        :param_id: 控制指令
        :value: 控制参数
        :return: 如果设置成功返回 True 否则返回 False
        """
        try:
            # 设置相机参数
            await asyncio.to_thread(
                set_camera_param, self.camera_addr, w_param_id, value
            )

            # logger.debug(f"set_camera_param w_param_id {value}")

        except Exception:
            self.logger.exception(f"设置参数 {w_param_id} 时发生错误")
            return False
        return True

    async def v4l2_w_and_r(self, param_id: int, value: int) -> bool:
        """
        发送v4l2可读可写指令: 探测器-曝光时间、模拟增益; 镜头：速度
        :param_id: 控制指令
        :value: 控制参数
        :return: 如果设置成功返回 True 否则返回 False
        """
        # —— 配置映射：用于状态回写（update_registers）——
        param_mapping = {
            v4l2_cmd["CTRL_EXPOSURE"]: "integ_time_setting",  # 以“时间(µs)”写回
            v4l2_cmd["CTRL_ANALOGUE_GAIN"]: "gain_setting",  # 以“实际读回值”写回
        }

        raw_value_to_set = int(value)

        # —— 若是曝光：先将“时间 t”转换为“设备 N 值” ——
        if param_id == v4l2_cmd["CTRL_EXPOSURE"]:
            try:
                raw_value_to_set = exposure_tool_t2n(
                    raw_value_to_set, self.wave, self.camera_identity
                )
            except Exception:
                self.logger.exception(
                    "曝光时间 t→N 转换失败: value=%s, wave=%s, cam=%s",
                    value,
                    getattr(self, "wave", None),
                    getattr(self, "camera_identity", None),
                )
                return False

        # —— SWIR 定制规则：设置前 N/=100，读回后再 *100 ——
        if self.camera_identity == "swir_fix" and param_id == v4l2_cmd["CTRL_EXPOSURE"]:
            scaled = int(raw_value_to_set / 100)
            self.logger.debug("SWIR: 设置前 N/=100, %s -> %s", raw_value_to_set, scaled)
            raw_value_to_set = scaled

        # —— 下发设置 ——
        try:
            await asyncio.to_thread(
                set_camera_param, self.camera_addr, param_id, raw_value_to_set
            )
        except Exception:
            self.logger.exception(
                "设置参数失败: param_id=%s, value=%s, addr=%s",
                param_id,
                raw_value_to_set,
                getattr(self, "camera_addr", None),
            )
            return False

        # —— 等待设置生效（保持原有 3s 语义；如需自定义可在实例上加 wait_after_set 属性）——
        wait_seconds = getattr(self, "wait_after_set", 3)
        await asyncio.sleep(wait_seconds)

        # —— 读回校验 ——
        try:
            re_value = await asyncio.to_thread(
                get_camera_param, self.camera_addr, param_id
            )
        except Exception:
            self.logger.exception(
                "读取参数失败: param_id=%s, addr=%s",
                param_id,
                getattr(self, "camera_addr", None),
            )
            return False

        self.logger.debug("v4l2_get_param param_id=%s, raw=%s", param_id, re_value)

        # —— 若是曝光：将读回的 N 值转换回“时间(µs)”并进行红外侧效应处理 ——
        exposure_time_us = None
        if param_id == v4l2_cmd["CTRL_EXPOSURE"]:
            # SWIR：读回后 *100 还原
            if self.camera_identity == "swir_fix":
                self.logger.debug(
                    "SWIR: 读回后 N*=100, %s -> %s", re_value, int(re_value) * 100
                )
                re_value = int(re_value) * 100

            try:
                exposure_time_us = exposure_tool_n2t(
                    re_value, self.wave, self.camera_identity
                )
            except Exception:
                self.logger.exception(
                    "曝光 N→t 转换失败: N=%s, wave=%s, cam=%s",
                    re_value,
                    getattr(self, "wave", None),
                    getattr(self, "camera_identity", None),
                )
                return False

            self.logger.debug("返回积分时间设置结果: %s 微秒", exposure_time_us)

            # 红外相机：通知图像处理器更新积分时间校正文件
            if self.wave in ["swir", "mwir", "lwir"]:
                temp_c = self.tec_temp if self.wave == "swir" else 0
                try:
                    temp_int = pack_temp_integration(
                        temp_c=temp_c, t_ms=exposure_time_us / 1000.0
                    )
                    self.set_integration_time(temp_int)
                except Exception:
                    self.logger.exception(
                        "更新积分时间校正失败: temp_c=%s, t_ms=%s",
                        temp_c,
                        exposure_time_us / 1000.0,
                    )
                    return False

        # —— 状态回写（寄存器/状态缓存）——
        if param_id in param_mapping:
            status_key = param_mapping[param_id]
            # 曝光：写“时间(µs)”；增益：写设备实际读回值
            status_val = (
                exposure_time_us if param_id == v4l2_cmd["CTRL_EXPOSURE"] else re_value
            )
            try:
                update_registers({status_key: status_val}, self.camera_identity)
                self.logger.debug(
                    "状态更新: %s=%s (cam=%s)",
                    status_key,
                    status_val,
                    self.camera_identity,
                )
            except Exception:
                self.logger.exception(
                    "状态更新失败: key=%s, val=%s, cam=%s",
                    status_key,
                    status_val,
                    self.camera_identity,
                )
                return False

        return True

    async def v4l2_r(self, param_id: int) -> bool:
        """
        发送v4l2查询指令: 探测器-曝光时间、模拟增益、绝对位置
        :param_id: 控制指令
        :value: 控制参数
        :return: 如果设置成功返回 True 否则返回 False
        """
        try:
            # 查询当前参数值
            re_value = await asyncio.to_thread(
                get_camera_param, self.camera_addr, param_id
            )
        except Exception as e:
            import traceback

            self.logger.error(
                f"读取状态 {param_id} 时发生错误: {traceback.print_exc()}"
            )
            time.sleep(3)
            return False

        param_mapping = {
            v4l2_cmd["CTRL_EXPOSURE"]: "integ_time_setting",
            v4l2_cmd["CTRL_ANALOGUE_GAIN"]: "gain_setting",
            v4l2_cmd["CTRL_FOCUS_CURRENT"]: "focus_abs_position",
            v4l2_cmd["CTRL_ZOOM_CURRENT"]: "hfov",
            v4l2_cmd["CTRL_TEMPERATURE_CURRENT"]: "camera_temperature",
            v4l2_cmd["CTRL_CURRENT_TEC"]: "camera_tec_current",
        }

        status_key = param_mapping.get(param_id)

        if param_id == v4l2_cmd["CTRL_EXPOSURE"]:

            self.logger.debug(f"查询波段：{self.wave} 到积分时间N值：{re_value}")

            if self.camera_identity == "swir_fix":
                self.logger.debug(f"短波积分时间乘以100哦！！！")
                re_value = int(re_value * 100)

            re_value = exposure_tool_n2t(re_value, self.wave, self.camera_identity)

            if self.wave in ["swir", "mwir", "lwir"]:

                # 如果是曝光时间且是红外相机，通知图像处理器更新积分时间校正文件
                temp = self.tec_temp if self.wave == "swir" else 0

                # note! 将微秒转换为毫秒
                temp_int = pack_temp_integration(temp_c=temp, t_ms=re_value / 1000)
                self.set_integration_time(temp_int)

        if param_id == v4l2_cmd["CTRL_ZOOM_CURRENT"]:
            self.logger.debug(f"读取视场角原始值：{re_value}")
            vol2focus = self._vol2focus(re_value)
            focus2hfov = self._focus2hfov(vol2focus)
            re_value = focus2hfov

        status = {status_key: int(re_value)} if status_key else {}

        try:
            update_registers(status, self.camera_identity)
        except Exception as e:
            self.logger.error(f"更新状态出错: {status_key} {e}")

        return True
