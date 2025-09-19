"""
1.红外图像
非均匀线性校正
盲元检测与补偿
直方图拉伸
引导滤波
2.可见光图像
白平衡校正
锐化
引导滤波
"""

import os
import numpy as np
import cv2
import logging
from numpy.lib.stride_tricks import sliding_window_view
import yaml


# 默认日志记录器，会被具体的相机模式日志记录器替换
logger = logging.getLogger(__name__)


# 全局日志记录器设置函数
def set_preutils_logger(camera_logger):
    """设置preutils模块的日志记录器"""
    global logger
    logger = camera_logger


current_dir = os.path.dirname(os.path.abspath(__file__))

TOL_factor = 0.01
pixel_type = np.uint16
pixel_max = 16383
PIXELTYPE = 4095  # 12位图像

# BK = cv2.imread("/userdata/camera-ctrl/camera/imager/004.png",-1)


class ImagePreprocessor:
    def __init__(self, npz_path: str = None):

        self.bp_para = None
        self.nuc_para = None
        self.blind_mask = None
        self.bp_path = None
        self.npz_path = None
        self.mid_it = None
        self.mid_t = None
        self.nuc_cfg_path = None

        npz_file_name = os.path.basename(npz_path)[:-5]

        self.wave = npz_file_name.split("_")[0]
        self.max_val = 16383 if self.wave == "lwir" else 4095

        if npz_file_name in ["lwir_fix_corr", "mwir_fix_corr", "swir_fix_corr"]:
            self.npz_path = npz_path
            self.load_config(self.npz_path)

        self.cur_temp_c_t = 0  # 当前温度
        self.cur_t_ms = 0  # 当前积分时间
        self.gain_map = None
        self.offset_map = None
        self.ref = None
        self.a_map = None
        self.b_map = None
        self.global_a = None
        self.global_b = None

    def _extract_correction_params(self):
        if self.nuc_para is None:
            return

        try:
            # 提取线性校正参数
            if all(key in self.nuc_para for key in ["a_map", "b_map", "ga", "gb"]):
                self.a_map = self.nuc_para["a_map"]
                self.b_map = self.nuc_para["b_map"]
                self.global_a = self.nuc_para["ga"]
                self.global_b = self.nuc_para["gb"]

            # 提取二次校正参数
            if all(key in self.nuc_para for key in ["a2", "a1", "a0"]):
                self.a2 = self.nuc_para["a2"]
                self.a1 = self.nuc_para["a1"]
                self.a0 = self.nuc_para["a0"]

            # 提取明暗校正参数
            if all(key in self.nuc_para for key in ["gain_map", "offset_map", "ref"]):
                self.gain_map = self.nuc_para["gain_map"]
                self.offset_map = self.nuc_para["offset_map"]
                self.ref = self.nuc_para["ref"]

        except Exception as e:
            logger.warning(f"提取校正参数时出错: {e}")
            # 如果提取失败，清空缓存
            self.a_map = None
            self.b_map = None
            self.global_a = None
            self.global_b = None
            self.a2 = None
            self.a1 = None
            self.a0 = None
            self.gain_map = None
            self.offset_map = None
            self.ref = None

    def load_config(self, npz_path):

        if not os.path.exists(npz_path):
            logger.error(f"文件 {npz_path} 不存在。")
            return {}

        try:
            with open(npz_path, "r", encoding="utf-8") as f:
                nuc_cfg_path = yaml.safe_load(f)

        except Exception as e:
            logger.warning(f"ImagePreprocessor load failed: {e}")
            return

        logger.debug(f"成功载入校正文件目录: {npz_path}")
        # print(nuc_cfg_path.keys())
        self.nuc_cfg_path = nuc_cfg_path

        self.default_load_correction_params(nuc_cfg_path)

    @staticmethod
    def parse_ms(k: str) -> float:
        return float(k.rstrip("ms"))

    def default_load_correction_params(self, nuc_cfg_path):

        subcfg = nuc_cfg_path.get("5t", {})
        temp_c = "5"
        # print(subcfg.keys())

        if not subcfg:
            # 说明是中长波：
            subcfg = nuc_cfg_path
            temp_c = 0
        # print(subcfg)
        its = sorted(subcfg.keys(), key=self.parse_ms)
        mid_it = its[len(its) // 2]  # 中位数位置

        logger.debug(f"默认加载温度 {temp_c}t 下 积分时间 {mid_it} 的校正文件.")
        # print(f"默认加载温度 {temp_c} 下 积分时间 {mid_it} 的校正文件.")

        return self.load_correction_params(temp_c=temp_c, t_ms=mid_it)

    @staticmethod
    def load_blind_npz(npz_path):
        if not os.path.exists(npz_path):
            logger.error(f"blind 文件{npz_path} 不存在。")
            return np.array([[]]), False

        # 加载npz文件
        data = np.load(npz_path)
        logger.debug(f"成功加载坏点校正文件: {npz_path}")

        bp_para = data["blind"].astype(bool)

        # return bp_para[::-1, :], True
        return bp_para, True

    @staticmethod
    def load_nuc_npz(npz_path):
        if not os.path.exists(npz_path):
            logger.error(f"nuc 文件{npz_path} 不存在。")
            return np.array([[]]), False

        # 加载npz文件
        data = np.load(npz_path)
        logger.debug(f"成功加载非均匀校正文件: {npz_path}")

        return data, True

    def unpack_temp_integration(self, packed: int) -> tuple[int, float]:
        """
        从 uint16 中解包温度与积分时间
        返回 (temp_c, t_ms)
        """
        temp_c, t_ms = 0, 0.0

        if not (0 <= packed <= 0xFFFF):
            logger.error("val must be uint16")
            return None, None
        temp = (packed >> 11) & 0x1F  # 高 5 位
        if not temp == 0:
            temp_c = self.map_temp_c(temp)

        count = packed & 0x7FF  # 低 11 位
        t_ms = count * 0.01  # 由 0.01 ms 计数转回毫秒

        if self.wave == "lwir":
            t_ms = format(t_ms, ".2f")  # 转换为毫秒,长波保留两位小数
        elif self.wave == "swir":
            t_ms = round(t_ms, 1)  # 转换为毫秒,只保留一位小数
            if not t_ms == 0.5:
                t_ms = int(t_ms)
        elif self.wave == "mwir":
            t_ms = int(t_ms)  # 转换为毫秒,直接取整
        else:
            logger.error(f"未知波段: {self.wave}, 无法解包积分时间")
            return None, None

        return temp_c, t_ms

    @staticmethod
    def map_temp_c(temp_c: float) -> int:
        if temp_c < 7.5:
            return 5
        else:
            return 10

    def load_correction_params(
        self, temp_int: int = None, temp_c: int = None, t_ms: int = None
    ):
        if self.nuc_cfg_path is None:
            logger.warning("配置文件不存在！")
            return

        logger.debug(
            f"read {self.wave} temp_int:{temp_int} temp_c:{temp_c} t_ms:{t_ms}"
        )

        if temp_int is not None:

            temp_c, t_ms = self.unpack_temp_integration(temp_int)  # >> int
            if temp_c is None and t_ms is None:
                logger.warning("积分时间解析错误！")
                return

            # 增加文字
            t_ms = f"{t_ms}ms"
            temp_c = f"{temp_c}t"
            logger.debug(f"parse temp_c, t_ms result: {temp_c}, {t_ms}")
            self.cur_temp_c_t = temp_c
            self.cur_t_ms = t_ms

        elif temp_c is not None and t_ms is not None:
            self.cur_temp_c_t = temp_c
            self.cur_t_ms = t_ms
            pass

        if t_ms == 0 or t_ms == "0.0ms":
            logger.warning("!!! >> t_ms is 0")
            return

        if temp_c == "0t" or temp_c == 0:
            try:
                self.bp_path = self.nuc_cfg_path[f"{t_ms}"]["bp"]
                nuc_path = self.nuc_cfg_path[f"{t_ms}"]["nuc"]
            except Exception as e:
                logger.warning(f"读取校正文件错误: {e}")
                return
        else:
            if isinstance(temp_c, str):
                temp_c = int(temp_c[0])
            try:
                self.bp_path = self.nuc_cfg_path[f"{temp_c}t"][f"{t_ms}"]["bp"]
                nuc_path = self.nuc_cfg_path[f"{temp_c}t"][f"{t_ms}"]["nuc"]
            except Exception as e:
                import traceback

                # logger.warning(f"读取校正文件错误: {e} \n {traceback.print_exc()}")
                logger.warning(f"读取校正文件错误: {e}")

                return

        bp_para, ret = self.load_blind_npz(self.bp_path)
        if ret:
            self.bp_para = bp_para

        nuc_para, ret = self.load_nuc_npz(nuc_path)
        if ret:
            self.nuc_para = nuc_para
            self._extract_correction_params()

    def default_process(self, frame, identity):
        """
        默认处理流程：非均匀校正 -> 盲元检测与补偿 -> 拉伸 -> 锐化
        """

        if identity == "swir_fix":
            process_res = self.apply_dw_calibration(
                frame, self.gain_map, self.offset_map, self.ref, self.max_val
            )

        elif identity in ["lwir_fix", "mwir_fix"]:
            process_res = self.apply_linear_calibration(
                frame,
                self.a_map,
                self.b_map,
                self.global_a,
                self.global_b,
                self.max_val,
            )

        process_res = self.compensate_blind_pixels(process_res, self.bp_para)

        process_res = self.stretch_u16(process_res, self.max_val)

        process_res = self.apply_sharping(process_res)

        return process_res

    def apply_dw_calibration(self, frame, gain_map, offset_map, ref, max_val):
        if gain_map is None or offset_map is None or ref is None:
            logger.warning("校正参数未加载。")
            return frame

        frame = frame.astype(np.float32)
        corrected = gain_map * frame + offset_map
        corrected = np.clip(corrected, 0, ref)
        corrected = (corrected / ref) * max_val
        return np.rint(corrected).astype(np.uint16)

    def apply_linear_calibration(
        self, frame, a_map, b_map, global_a, global_b, max_val, eps_gain=1e-6
    ):
        if a_map is None or b_map is None or global_a is None or global_b is None:
            logger.warning("校正参数未加载。")
            return frame

        frame = frame.astype(np.float32)

        # 增益防护
        a_safe = np.where(
            np.abs(a_map) < eps_gain, np.sign(a_map) * eps_gain, a_map
        ).astype(np.float32)

        corr = (frame - b_map) / a_safe

        corr = corr * float(global_a) + float(global_b)

        corr = np.clip(corr, 0, float(max_val))
        return np.rint(corr).astype(np.uint16)

    def compensate_blind_pixels(self, frame, bp_para, ksize: int = 5):
        if bp_para is None:
            logger.warning("盲元校正参数未加载。")
            return frame

        if ksize < 3:
            ksize = 3
        if ksize % 2 == 0:
            ksize += 1

        blurred = cv2.medianBlur(frame, ksize)

        frame[bp_para] = blurred[bp_para]

        frame = frame.astype(np.uint16)

        return frame

    def stretch_u16(self, frame, max_val):

        q_low, q_high = 0.5, 99.5
        min_range = 0.03
        noise = 0.0005  # 0.0005–0.001
        adj_gamma = 0.9

        eps = 1e-6
        frame = frame.astype(np.float32)
        frame = cv2.medianBlur(frame, ksize=3)

        lo = float(np.percentile(frame, q_low))
        hi = float(np.percentile(frame, q_high))

        valid_range = max(hi - lo, max_val * min_range, eps)

        x = np.arange(max_val + 1, dtype=np.float32)
        y = (x - lo + max_val * noise) / valid_range
        np.clip(y, 0.0, 1.0, out=y)

        y = np.power(y, adj_gamma) * max_val

        midtone_boost = 0.08  # 建议 0.05~0.12，小幅提亮
        y_norm = y / max_val
        y_norm = y_norm + midtone_boost * y_norm * (1.0 - y_norm)  # 只抬中灰
        y = y_norm * max_val

        lut = np.clip(y, 0.0, float(max_val)).astype(np.uint16)

        frame_u16 = np.clip(frame, 0, max_val).astype(np.uint16)

        return lut[frame_u16]

    @staticmethod
    def scale_to_8bit(frame_16bit, max_val=65535):
        return cv2.convertScaleAbs(frame_16bit, alpha=255.0 / max_val, beta=0)

    def storge_blindpixels(self, blind_mask):
        """
        在原有盲元表的基础上叠加后再覆盖原有文件
        """
        if self.bp_para is None:
            logger.warning("bp_para 尚未加载。")
            return False

        self.bp_para = np.logical_or(self.bp_para, blind_mask).astype(
            self.bp_para.dtype
        )

        try:
            np.savez_compressed(self.bp_path, blind=self.bp_para)
        except Exception as e:
            logger.warning(f"保存盲元失败:{e}")
            return False
        return True

    def apply_blind_pixel_detect(self, frame, sigma=3, window_size=33):

        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rows, cols = frame.shape
        half = window_size // 2

        # 使用滑动窗口视图避免逐像素循环
        window_view = sliding_window_view(frame, (window_size, window_size))

        # 计算每个窗口的均值和标准差
        window_flat = window_view.reshape(-1, window_size * window_size)
        means = np.mean(window_flat, axis=1).reshape(rows - 2 * half, cols - 2 * half)
        stds = np.std(window_flat, axis=1).reshape(rows - 2 * half, cols - 2 * half)

        # 提取中心像素值
        centers = frame[half:-half, half:-half]

        # 生成盲元掩码
        mask = np.abs(centers - means) > sigma * stds
        blind_mask = np.zeros((rows, cols), dtype=np.bool_)
        blind_mask[half:-half, half:-half] = mask

        self.blind_mask = blind_mask
        return blind_mask

    def apply_sharping(self, image):
        blurred = cv2.GaussianBlur(image, (5, 5), 1.5)
        result = cv2.addWeighted(image, 2, blurred, -1.0, 0)
        return result

    def apply_autogian(self, frame):

        if frame.dtype not in [np.uint8, np.uint16]:
            if frame.max() > 255:
                frame = frame.astype(np.uint16)
            else:
                frame = frame.astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        result = clahe.apply(frame)

        return result

    def draw_center_cross_polylines(self, image, color=(255, 255, 255), thickness=5):
        """
        使用polylines高效绘制十字线
        :param cross_radius: 十字线半径（从中心到端点的距离）
        """
        height, width = image.shape[:2]
        center_x = width // 2
        center_y = height // 2

        cross_radius = int(width * 0.1)
        # print(cross_radius)

        # 定义横线和竖线的顶点坐标
        horizontal_line = np.array(
            [
                [[center_x - cross_radius, center_y]],
                [[center_x + cross_radius, center_y]],
            ],
            dtype=np.int32,
        )

        vertical_line = np.array(
            [
                [[center_x, center_y - cross_radius]],
                [[center_x, center_y + cross_radius]],
            ],
            dtype=np.int32,
        )

        # 绘制所有线段
        cv2.polylines(
            image,
            [horizontal_line, vertical_line],
            isClosed=False,
            color=color,
            thickness=thickness,
        )
        return image
