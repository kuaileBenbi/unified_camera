from status import set_values
import logging

# 默认日志记录器，会被具体的相机模式日志记录器替换
logger = logging.getLogger(__name__)


# 全局日志记录器设置函数
def set_updater_logger(camera_logger):
    """设置updater模块的日志记录器"""
    global logger
    logger = camera_logger


frame_wh = {
    "vis_zoom": (1024, 1024),
    "mwir_zoom": (256, 320),
    "vis_fix": (1024, 1024),
    "swir_fix": (320, 256),
    "mwir_fix": (320, 256),
    "lwir_fix": (320, 256),
}


class DetResUpdater:

    def __init__(self):
        self.dl_model_mode = 0
        self.tem_model_mode = 0

    def set_mode(self, dl_model_mode, tem_model_mode):
        self.dl_model_mode = dl_model_mode
        self.tem_model_mode = tem_model_mode

    def set_frame_mode(self, camera_tittle):
        # print(f"camera_tittle: {camera_tittle}")
        self.frame_cx, self.frame_cy = frame_wh.get(camera_tittle, (0, 0))

    def encode_status_to_uint8(
        self, mode, method, has_plane, has_ship, has_car, reserved=0
    ):
        """
        将多种状态组合到一个 UINT8 里，位定义如下:
        bit7-6: 模式 (mode)
            - 0: 不检测
            - 1: 只检测
            - 2: 检测 + 跟踪
            - 3: 预留

        bit5-4: 方法 (method)
            - 0: 单模型
            - 1: 双模型
            - 2: 固定模板
            - 3: 自适应模板

        bit3: 有无飞机 (has_plane)
            - 0: 无飞机
            - 1: 有飞机

        bit2: 有无舰船 (has_ship)
            - 0: 无舰船
            - 1: 有舰船

        bit1: 有无车辆 (has_car)
            - 0: 无车辆
            - 1: 有车辆

        bit0: 预留 (reserved)
        """
        # 确保各字段都在合理范围内
        mode &= 0b11  # mode仅2位
        method &= 0b11  # method仅2位
        has_plane &= 0b1
        has_ship &= 0b1
        has_car &= 0b1
        reserved &= 0b1

        # 构造字节：
        # 高位到低位依次：mode (2 bits), method (2 bits), has_plane, has_ship, has_vehicle, reserved
        status_value = 0
        status_value |= mode << 6
        status_value |= method << 4
        status_value |= has_plane << 3
        status_value |= has_ship << 2
        status_value |= has_car << 1
        status_value |= reserved << 0

        return status_value

    def update_status(self, detect_classes: list, detect_boxes: list):

        # print(f"dl_mode:{self.dl_model_mode}")
        # print(f"tem_mode: {self.tem_model_mode}")
        # print(f"wh: {self.frame_cx} {self.frame_cy}")

        if self.dl_model_mode == 0 and self.tem_model_mode == 0:
            logger.warning("未识别到正确的检测模式，直接返回！")
            return

        # print(f"detect_classes: {detect_classes}")
        # print(f"detect_boxes: {detect_boxes}")

        status = {}

        # 确定模式 (mode)
        mode_mapping = {1: 1, 2: 2}
        mode = mode_mapping.get(self.dl_model_mode, 0)

        # 确定方法 (method)
        if self.tem_model_mode in [1, 2]:
            # 1:固定模板 2:自适应模板
            method_mapping = {1: 2, 2: 3}
            method = method_mapping[self.tem_model_mode]
        else:
            # tem_model_mode=0 不开启模板匹配
            # 0: 单模型 1：双模型
            method_mapping = {1: 0, 2: 1}
            method = method_mapping.get(self.dl_model_mode, 0)

        # 提取检测的类别集
        detected_classes = set(detect_classes)

        # 设置有无飞机、舰船和车辆
        has_plane = 1 if "airplane" in detected_classes else 0
        has_ship = 1 if "boat" in detected_classes else 0
        has_car = 1 if "car" in detected_classes else 0

        # 预留位设为0
        reserved = 0

        status["detect_track_status1"] = self.encode_status_to_uint8(
            mode, method, has_plane, has_ship, has_car, reserved
        )
        # status["detect_result_count"] = len(detect_res.get("classes", []))
        status["detect_result_count"] = len(detect_classes)

        # 初始化目标类型，默认为0
        status["target1_4_type"] = 0
        status["target5_6_type"] = 0

        # 类别到编码的映射
        class_mapping = {
            "boat": 1,  # ship
            "airplane": 2,  # airplane
            "car": 3,  # vehicle
        }

        # 初始化目标1到目标6的中心坐标、宽度和高度
        for i in range(1, 7):
            status[f"target{i}_center_x"] = self.frame_cx
            status[f"target{i}_center_y"] = self.frame_cy
            status[f"target{i}_width"] = 0
            status[f"target{i}_height"] = 0

        # 使用并发处理每个目标的状态更新
        def update_target(i, box, class_name):
            # print(i)
            left, top, right, bottom = box
            center_x = int((left + right) / 2)
            center_y = int((top + bottom) / 2)
            width = int(right - left)
            height = int(bottom - top)

            target_idx = i + 1  # target1, target2, ...

            # 填充中心坐标、宽度和高度
            status[f"target{target_idx}_center_x"] = center_x
            status[f"target{target_idx}_center_y"] = center_y
            status[f"target{target_idx}_width"] = width
            status[f"target{target_idx}_height"] = height

            # 获取类别名称并映射到编码
            class_code = class_mapping.get(class_name, 0)

            # 根据目标索引设置类型编码
            if 1 <= target_idx <= 4:
                shift = (target_idx - 1) * 2
                status["target1_4_type"] |= (class_code & 0x03) << shift
            elif 5 <= target_idx <= 6:
                shift = (target_idx - 5) * 2
                status["target5_6_type"] |= (class_code & 0x03) << shift

        # 处理目标
        # tasks = [
        #     update_target(i, box, detect_res["classes"][i])
        #     for i, box in enumerate(detect_res["boxes"][:6])
        # ]
        tasks = [
            update_target(i, box, detect_classes[i])
            for i, box in enumerate(detect_boxes[:6])
        ]

        # print(status)
        set_values(status)
