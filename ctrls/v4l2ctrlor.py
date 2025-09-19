# Changelog
# v0.1 初始版本

import fcntl
from ctypes import *
import logging
import subprocess

# 默认日志记录器，会被具体的相机模式日志记录器替换
logger = logging.getLogger(__name__)


# 全局日志记录器设置函数
def set_v4l2_logger(camera_logger):
    """设置v4l2ctrlor模块的日志记录器"""
    global logger
    logger = camera_logger


v4l2_cmd = {
    "CTRL_EXPOSURE": 0x00980911,  # 曝光时间
    "CTRL_ZOOM_CONTINUOUS": 0x009819D0,  # 连续变焦
    "CTRL_FOCUS_CONTINUOUS": 0x009819D1,  # 连续对焦
    "CTRL_ISIS_CONTINUOUS": 0x009819D2,  # 连续光圈
    "CTRL_STOP": 0x009819D3,  # 停止变焦/对焦
    "CTRL_FOCUS_JOG": 0x009819D4,  # 对焦摇杆控制
    "CTRL_ZOOM_ABSOLUTE": 0x009819D5,  # 绝对变焦
    "CTRL_FOCUS_ABSOLUTE": 0x009819D6,  # 绝对对焦
    "CTRL_FOCUS_SPEED": 0x009819D7,  # 对焦速度
    "CTRL_ZOOM_CURRENT": 0x009819D8,  # 当前变焦值
    "CTRL_FOCUS_CURRENT": 0x009819D9,  # 当前对焦值
    "CTRL_IRIS_CURRENT": 0x009819DA,  # 当前光圈值
    "CTRL_ANALOGUE_GAIN": 0x009E0903,  # 模拟增益
    "CTRL_TEMPERATURE_CURRENT": 0x009819DB,  # 当前温度
    "CTRL_TEMPERATURE_TEC": 0x009819DC,  # TEC温度
    "CTRL_CURRENT_TEC": 0x009819DD,  # 当前TEC值
    "VIDIOC_S_CTRL": 0xC0144809,  # V4L2控件请求码
}


# 定义V4L2控制结构体
class v4l2_control(Structure):
    _fields_ = [("id", c_uint32), ("value", c_int32)]


def send_v4l2_stop_command(device):
    try:
        cmd = f"v4l2-ctl -d {device} --set-ctrl=stop=1"
        subprocess.run(cmd, shell=True, check=True)
        logger.debug("命令执行成功")
    except subprocess.CalledProcessError as e:
        logger.error(f"命令执行失败: {e}")


def send_v4l2_focus_absolute_command(device, value=10000):
    try:
        cmd = f"v4l2-ctl -d {device} --set-ctrl=focus_absolute={value}"
        subprocess.run(cmd, shell=True, check=True)
        logger.debug("命令执行成功")
    except subprocess.CalledProcessError as e:
        logger.error(f"命令执行失败: {e}")


def send_v4l2_vis_command():
    try:
        # "i2ctransfer -y -f 2 w5@0x28 0x00 0x64 0x01 0x00 0x00"
        cmd = "i2ctransfer -y -f 9 w5@0x28 0x00 0x64 0x01 0x00 0x00"
        subprocess.run(cmd, shell=True, check=True)
        logger.debug("可见变焦命令执行成功")
    except subprocess.CalledProcessError as e:
        logger.error(f"可见变焦命令执行失败: {e}")


def send_v4l2_vis_gain_command(device, value=0):
    try:
        cmd = f"v4l2-ctl -d {device} --set-ctrl=analogue_gain={value}"
        subprocess.run(cmd, shell=True, check=True)
        logger.debug("可见变焦增益命令执行成功")
    except subprocess.CalledProcessError as e:
        logger.error(f"可见变焦增益命令执行失败: {e}")


def calculate_cropped_size(original_width, original_height, top, bottom, left, right):
    width = original_width - (left + right)
    height = original_height - (top + bottom)
    return width, height


def set_camera_param(device_path, param_id, value):
    fd = open(device_path, "rb+", buffering=0)
    ctrl = v4l2_control()
    ctrl.id = param_id
    ctrl.value = value
    try:
        fcntl.ioctl(fd, 0xC008561C, ctrl)  # VIDIOC_S_CTRL
    finally:
        fd.close()


def get_camera_param(device_path, param_id):
    fd = open(device_path, "rb+", buffering=0)
    ctrl = v4l2_control()
    ctrl.id = param_id
    try:
        fcntl.ioctl(fd, 0xC008561B, ctrl)  # VIDIOC_G_CTRL
        return ctrl.value
    finally:
        fd.close()


def integration_tool(wave, re_value):
    if wave == "swir":
        return n_to_integration_time_infra(re_value, tmck_hz=10e6)
    elif wave == "lwir" or wave == "mwir":
        return n_to_integration_time_infra(re_value, tmck_hz=5e6)


def integration_time_to_n_4k(
    integration_time_us: int,
    *,
    total_lines: int = 4375,
    reserved_lines: int = 2,
    line_period_us: float = 9.14,
    n_max: int = 4000,
) -> int:
    """
    根据4K协议将积分时间（微秒）转换为寄存器GRSTW[23:0]的值N。

    参数
    ----
    integration_time_us : int
        积分时间，单位微秒
    total_lines : int
        总行数（包括保留行），默认4375
    reserved_lines : int
        保留行数，不计入积分，默认2
    line_period_us : float
        每行周期，单位微秒，默认9.14µs
    n_max : int
        N的上限，默认4000

    返回
    ----
    int
        对应的寄存器值N

    异常
    ----
    ValueError
        当计算出的N不在[0, n_max]范围内时抛出
    """
    # 计算有效行数
    effective_lines = round(integration_time_us / line_period_us)

    # 计算N值
    N = total_lines - reserved_lines - effective_lines

    # 检查N范围
    if not (0 <= N <= n_max):
        logger.error(f"计算出的N值{N}超出允许范围[0, {n_max}], 返回默认值200")
        return total_lines + reserved_lines - 200

    # 转换为用户设置的n_user值
    n_user = total_lines + reserved_lines - N

    return int(n_user)


def n_to_integration_time_4k(
    n_user: int,
    *,
    total_lines: int = 4375,
    reserved_lines: int = 2,
    line_period_us: float = 9.14,
    n_max: int = 4000,
) -> int:
    """
    根据 4K 协议将寄存器 GRSTW[23:0] 的值 N 转换为积分时间（微秒）。

    参数
    ----
    n : int
        寄存器值 N，范围 0 ~ n_max，默认 0x01。
    total_lines : int
        总行数（包括保留行），默认 4375。
    reserved_lines : int
        保留行数，不计入积分，默认 2。
    line_period_us : float
        每行周期，单位微秒，默认 9.14 µs。
    n_max : int
        N 的上限，默认 4000。

    返回
    ----
    float
        对应的积分时间，单位微秒。

    异常
    ----
    ValueError
        当 n 不在 [0, n_max] 范围内，或计算出的有效行数为负时抛出。
    """

    # 映射回协议里的 N
    N = total_lines + reserved_lines - n_user

    # 检查 N 范围
    if not (0 <= N <= n_max):
        logger.warning(f"N 的取值应在 0~{n_max} 之间（当前 {N}）")
        return 0

    # 计算有效积分行数
    effective_lines = total_lines - reserved_lines - N
    if effective_lines < 0:
        logger.warning(f"有效行数 = {effective_lines}，小于 0，请检查 N 和行数设置")
        return 0

    # 计算积分时间（微秒）并转换为秒
    integration_time_us = effective_lines * line_period_us
    return int(integration_time_us)


def integration_time_to_n_2k(
    integration_time_us: int,
    *,
    total_lines: int = 2100,
    reserved_lines: int = 2,
    line_period_us: float = 15.87,
    n_max: int = 2000,
) -> int:
    """
    根据2K协议将积分时间（微秒）转换为寄存器GRSTW[23:0]的值N。

    参数
    ----
    integration_time_us : int
        积分时间，单位微秒
    total_lines : int
        总行数（包括保留行），默认2100
    reserved_lines : int
        保留行数，默认2
    line_period_us : float
        每行周期，单位微秒，默认15.87µs
    n_max : int
        N的上限，默认2000

    返回
    ----
    int
        对应的寄存器值N

    异常
    ----
    ValueError
        当计算出的N不在[0, n_max]范围内时抛出
    """
    effective_lines = round(integration_time_us / line_period_us)
    N = total_lines - reserved_lines - effective_lines

    if not (0 <= N <= n_max):
        logger.error(f"计算出的N值{N}超出允许范围[0, {n_max}], 返回默认值200")
        return total_lines + reserved_lines - 200

    n_user = total_lines + reserved_lines - N
    return int(n_user)


def n_to_integration_time_2k(
    n_user: int,
    *,
    total_lines: int = 2100,
    reserved_lines: int = 2,
    line_period_us: float = 15.87,
    n_max: int = 2000,
) -> int:
    """
    根据 2K 协议将寄存器 GRSTW[23:0] 的值 N 转换为积分时间（微秒）。

    参数
    ----
    n : int
        寄存器值 N，范围 0 ~ n_max，默认 0x01。
    total_lines : int
        总行数（包括保留行），默认 2100。
    reserved_lines : int
        保留行数，不计入积分，默认 2。
    line_period_us : float
        每行周期，单位微秒，默认 15.87 µs。
    n_max : int
        N 的上限，默认 2000。

    返回
    ----
    float
        对应的积分时间，单位微秒。

    异常
    ----
    ValueError
        当 n 不在 [0, n_max] 范围内时抛出。
    """

    # 先映射回协议里的 N
    N = total_lines + reserved_lines - n_user

    # 校验映射后的 N
    if not (0 <= N <= n_max):
        logger.warning(f"N 的取值应在 0~{n_max} 之间（当前 {N}）")
        return 0

    effective_lines = total_lines - reserved_lines - N
    if effective_lines < 0:
        logger.warning(
            f"计算得到的有效行数为负（{effective_lines}），请检查 N 和行数设置"
        )
        return 0

    integration_time_us = effective_lines * line_period_us
    return int(integration_time_us)


def integration_time_to_n_infra(integration_time_us: int, tmck_hz: float = 10e6) -> int:
    """
    根据协议将积分时间（微秒）转换为积分时间参数N。

    参数：
        integration_time_us (int): 积分时间，单位微秒
        tmck_hz (float): 时钟频率，单位Hz，默认为10e6（即10MHz）

    返回：
        int: 对应的积分时间参数N

    异常：
        ValueError: 当计算出的N不在[400, 210000]范围内时抛出
    """
    # 将微秒转换为秒
    integration_time_s = integration_time_us / 1e6

    # 计算N值
    n = round(integration_time_s * tmck_hz)

    # 校验N范围
    if not (400 <= n <= 210_000):
        logger.error(f"计算出的N值{n}超出允许范围[400, 210000],返回默认值50000")
        return 50000

    return int(n)


def n_to_integration_time_infra(n: int = 1000, tmck_hz: float = 10e6) -> int:
    """
    根据协议将积分时间参数 N 转换为积分时间（微秒）。

    参数：
        n (int): 积分时间参数，取值范围 [400, 210000]，默认 1000。
        tmck_hz (float): 时钟频率，单位 Hz，默认为 10e6（即 10 MHz）。

    返回：
        float: 对应的积分时间，单位为微秒。

    异常：
        ValueError: 当 n 不在 [400, 210000] 范围内时抛出。
    """
    # 校验 N 的范围
    if not (400 <= n <= 210_000):
        logger.warning(f"N 的取值应在 400~210000 之间（当前 {n}）")
        return 0.0

    # 时钟周期（秒）
    tmck_period = 1.0 / tmck_hz

    # 积分时间 = N × 时钟周期
    integration_time_s = n * tmck_period
    return int(integration_time_s * 1e6)  # 返回微秒


def pack_temp_integration_lsb_0_1(temp_c: int, t_ms: float) -> int:
    """
    把温度和积分时间打包到 uint16
    - 高 8 位 temp_c (0-255)
    - 低 8 位 round(t_ms / 0.1) (0-255)
    """
    # 限定范围
    if not (0 <= temp_c <= 0xFF):
        logger.warning("temp_c must be 0–255")
        return 0
    count = t_ms * 10  # 转换为 0.1 ms 单位
    if not (0 <= count <= 0xFF):
        logger.warning("t_ms out of range (0.0–25.5ms)")
        return 0
    return (temp_c << 8) | (int(count) & 0xFF)


def pack_temp_integration(temp_c: int, t_ms: float) -> int:
    """
    把温度与积分时间打包成 uint16
    - 高 5 位：温度 temp_c(0-31 ℃）
    - 低 11 位: round(t_ms / 0.01) 0-2047, 对应 0-20.47 ms
    """
    if not (0 <= temp_c <= 31):
        logger.warning("temp_c must be 0-31")
        return 0

    count = int(round(t_ms * 100))  # 0.01 ms 为单位
    if not (0 <= count <= 2047):  # 11 bit 范围
        logger.warning("t_ms out of range (0–20.47 ms)")
        return 0

    return ((temp_c & 0x1F) << 11) | (count & 0x7FF)


def integration_tool_n2t(wave, re_value):
    if wave == "swir":
        return n_to_integration_time_infra(re_value, tmck_hz=10e6)
    elif wave == "lwir" or wave == "mwir":
        return n_to_integration_time_infra(re_value, tmck_hz=5e6)


def integration_tool_t2n(wave, re_value):
    if wave == "swir":
        return integration_time_to_n_infra(re_value, tmck_hz=10e6)
    elif wave == "lwir" or wave == "mwir":
        return integration_time_to_n_infra(re_value, tmck_hz=5e6)


def exposure_tool_n2t(re_value: int, wave: str, camera_identity: str) -> int:
    if camera_identity in ["swir_fix", "mwir_fix", "lwir_fix"]:
        return integration_tool_n2t(wave, re_value)  # 微秒
    elif camera_identity == "vis_zoom":
        return n_to_integration_time_2k(re_value)
    elif camera_identity == "vis_fix":
        return n_to_integration_time_4k(re_value)
    else:
        logger.warning("不存在的积分时间转换！")
        return 0


def exposure_tool_t2n(integration_time_us: int, wave: str, camera_identity: str) -> int:
    """
    根据相机类型将积分时间(微秒)转换为寄存器值N

    参数：
        integration_time_us: 积分时间(微秒)
        wave: 波段信息
        camera_identity: 相机标识

    返回：
        int: 寄存器值N

    异常：
        ValueError: 当计算出的N值超出范围时抛出
    """
    if camera_identity in ["swir_fix", "mwir_fix", "lwir_fix"]:
        return integration_tool_t2n(wave, integration_time_us)
    elif camera_identity == "vis_zoom":
        return integration_time_to_n_2k(integration_time_us)
    elif camera_identity == "vis_fix":
        return integration_time_to_n_4k(integration_time_us)
    else:
        logger.warning("不存在的积分时间转换！")
        return 0


if __name__ == "__main__":
    #     set_camera_param("/dev/video11", CTRL_EXPOSURE, 300)
    print(get_camera_param("/dev/video22", v4l2_cmd["CTRL_TEMPERATURE_CURRENT"]))
