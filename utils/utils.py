import logging

logger = logging.getLogger(__name__)

measure_map_once = {
    "vis_zoom": [
        "CTRL_EXPOSURE",
        "CTRL_ANALOGUE_GAIN",
        "CTRL_FOCUS_CURRENT",
        "CTRL_ZOOM_CURRENT",
    ],  # 曝光时间/视场角/对焦绝对值/增益
    "vis_fix": [
        "CTRL_EXPOSURE",
        "CTRL_ANALOGUE_GAIN",
        "CTRL_FOCUS_CURRENT",
    ],  # 曝光时间/对焦绝对值/增益
    "mwir_fix": [
        "CTRL_EXPOSURE",
        "CTRL_FOCUS_CURRENT",
        "CTRL_TEMPERATURE_CURRENT",
    ],  # 曝光时间/对焦绝对值/相机温度
    "swir_fix": [
        "CTRL_EXPOSURE",
        "CTRL_FOCUS_CURRENT",
        "CTRL_TEMPERATURE_CURRENT",
        "CTRL_CURRENT_TEC",
    ],  # 曝光时间/对焦绝对值/相机温度/tec电流
    "lwir_fix": [
        "CTRL_EXPOSURE",
        "CTRL_FOCUS_CURRENT",
        "CTRL_TEMPERATURE_CURRENT",
    ],  # 曝光时间/对焦绝对值/相机温度
}

measure_map_sched = {
    "vis_zoom": [
        "CTRL_FOCUS_CURRENT",
        "CTRL_ZOOM_CURRENT",
    ],  # 视场角/对焦绝对值
    "vis_fix": [
        "CTRL_FOCUS_CURRENT",
    ],  # 对焦绝对值
    "mwir_fix": [
        "CTRL_FOCUS_CURRENT",
        "CTRL_TEMPERATURE_CURRENT",
    ],  # 对焦绝对值/相机温度
    "swir_fix": [
        "CTRL_FOCUS_CURRENT",
        "CTRL_TEMPERATURE_CURRENT",
        "CTRL_CURRENT_TEC",
    ],  # 对焦绝对值/相机温度/tec电流
    "lwir_fix": [
        "CTRL_FOCUS_CURRENT",
        "CTRL_TEMPERATURE_CURRENT",
    ],  # 对焦绝对值/相机温度
}


def restore_xywh(bbox, W, H, mode):
    """
    bbox: [cx, cy, w, h] —— 翻转/旋转后的中心点 + 宽高
    W, H: 原图尺寸
    mode: 'flip_h' | 'flip_v' | 'flip_hv' | 'rot90ccw'

    返回: [x0, y0, w0, h0] —— 原图左上角 + 宽高
    """
    cx, cy, w, h = bbox

    if mode == "flip_h":
        cx0, cy0 = W - 1 - cx, cy
        w0, h0 = w, h

    elif mode == "flip_v":
        cx0, cy0 = cx, H - 1 - cy
        w0, h0 = w, h

    elif mode == "flip_hv":
        cx0, cy0 = W - 1 - cx, H - 1 - cy
        w0, h0 = w, h

    elif mode == "rot90ccw":
        # 旋转后的图像宽高
        W_rot, H_rot = H, W
        # 中心点逆变换
        cx0 = W - 1 - cy
        cy0 = cx
        # 宽高互换
        w0, h0 = h, w

    else:
        raise ValueError(f"Unsupported mode {mode}")

    # 从中心点转成左上角
    x0 = int(round(cx0 - w0 / 2))
    y0 = int(round(cy0 - h0 / 2))

    return [x0, y0, int(w0), int(h0)]


def restore_cxcywh(bbox, W, H, mode):
    """
    bbox: [cx, cy, w, h] 把翻转后 (cx, cy, w, h) 映射回原图坐标。
    mode: 'flip_h'|'flip_v'|'flip_hv'|'rot90ccw'

    """

    [cx, cy, w, h] = bbox
    if mode == "flip_h":
        cx0, cy0 = W - 1 - cx, cy
        w0, h0 = w, h
    elif mode == "flip_v":
        cx0, cy0 = cx, H - 1 - cy
        w0, h0 = w, h
    elif mode == "flip_hv":
        cx0, cy0 = W - 1 - cx, H - 1 - cy
        w0, h0 = w, h
    elif mode == "rot90ccw":
        # 逆时针 90°：cx₀ = W-1-cy′；cy₀ = cx′；宽高互换
        cx0, cy0 = W - 1 - cy, cx
        w0, h0 = h, w
    else:
        raise ValueError(f"Unsupported mode {mode}")

    return [cx0, cy0, w0, h0]


def map_boxes_to_crop(cxcywh_box, full_size=(2048, 2048), crop_size=640):
    """
    boxes: List of (x1,y1,x2,y2) in full-resolution coordinates
    full_size: (W_full, H_full)
    crop_size: size of the centered square crop
    返回: List of (x1',y1',x2',y2') 映射到 crop 内的坐标，并自动裁剪到 [0,crop_size]
    """
    """
    将全图(center_x, center_y, width, height) 映射到中心 crop_size×crop_size 区域，
    并裁剪到 crop 边界内。

    Args:
        cxcywh_box: List of (cx, cy, w, h) in 全图坐标
        full_size: (W_full, H_full)
        crop_size: 裁剪边长

    Returns:
        List of (cx', cy', w', h')，坐标均相对 crop 左上角(0,0)，
        且只返回与 crop 有交集的框。
    """
    W_full, H_full = full_size
    dx = (W_full - crop_size) // 2
    dy = (H_full - crop_size) // 2

    [cx, cy, w, h] = cxcywh_box

    # 1) 计算原框的角点
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    # 2) 平移到 crop 坐标系
    nx1, ny1 = x1 - dx, y1 - dy
    nx2, ny2 = x2 - dx, y2 - dy

    # 3) 剔除完全在 crop 之外的框
    if nx2 <= 0 or ny2 <= 0 or nx1 >= crop_size or ny1 >= crop_size:
        return []

    # 4) 在 crop 内截断
    nx1 = max(0, min(crop_size, nx1))
    ny1 = max(0, min(crop_size, ny1))
    nx2 = max(0, min(crop_size, nx2))
    ny2 = max(0, min(crop_size, ny2))

    # 5) 转回 (cx, cy, w, h)
    ncx = (nx1 + nx2) / 2
    ncy = (ny1 + ny2) / 2
    nw = nx2 - nx1
    nh = ny2 - ny1

    # 如果截断后框的面积有效，才保留
    if nw > 0 and nh > 0:
        return [ncx, ncy, nw, nh]

    return []


def flip_down_up(x, y, W, H):
    """下上翻转（即上下翻转）"""
    return x, H - 1 - y


def flip_right_left(x, y, W, H):
    """右左翻转（即左右翻转）"""
    return W - 1 - x, y


def flip_down_up_right_left(x, y, W, H):
    """下上+右左翻转（即上下+左右翻转）"""
    return W - 1 - x, H - 1 - y


def rotate_90_cw(x, y, W, H):
    """
    顺时针旋转90°后，图像尺寸变为 (H, W)。
    返回新坐标 (x', y')，其中 0 <= x' < H, 0 <= y' < W。
    """
    return H - 1 - y, x


def estimated_image_size(w, h, codec="jpg", quality=85):
    mp = w * h / 1_000_000  # 百万像素
    k = {"jpg": 0.8, "png": 4.0, "bmp": 9.0}[codec]  # 经验值
    if codec == "jpg":
        k *= (quality / 85.0) ** 0.6  # 粗略修正
    return k * mp  # MB


def estimated_size_h264(bitrate_bps, segment_duration_s):
    # bitrate_bps: bits per second
    # segment_duration_s: seconds
    bytes_total = bitrate_bps * segment_duration_s / 8  # 字节
    bytes_total *= 1.05  # 加 5% 封装开销
    return bytes_total / (1024 * 1024)  # 转成 MB


def ip_to_uint32(ip_str):
    """将点分十进制IP字符串转换为32位无符号整数"""
    octets = list(map(int, ip_str.split(".")))
    if len(octets) != 4 or any(octet > 255 for octet in octets):
        raise ValueError("Invalid IPv4 address format")
    return (octets[0] << 24) | (octets[1] << 16) | (octets[2] << 8) | octets[3]


def decimal_to_hex(number):
    """
    将十进制数转换为十六进制字符串
    格式：两位低字节为小数，其余字节为整数
    """
    # 分离整数和小数部分
    integer_part = int(number)
    fractional_part = int(round((number - integer_part) * 256))  # 0-255范围

    # 处理整数部分
    hex_integer = hex(integer_part)[2:]  # 去掉0x前缀

    # 处理小数部分（确保是两位十六进制）
    hex_fraction = hex(fractional_part)[2:].zfill(2)

    # 组合结果
    result = hex_integer + hex_fraction
    return result.upper()  # 返回大写形式


def hex_to_decimal_with_fraction(n):
    """
    将整数转换为带两位小数的十进制数
    规则：十六进制表示的最后两个字节作为小数部分

    参数:
        n (int): 输入的整数

    返回:
        float: 组合后的带1位小数的十进制数
    """
    # print(f"输入整数: {n}")
    hex_str = hex(n)[2:].zfill(4)  # 转换为至少4位的十六进制字符串

    # 分割整数和小数部分
    int_part_hex = hex_str[:-4] if len(hex_str) > 4 else "0"
    frac_part_hex = hex_str[-4:]

    # 转换为十进制
    int_value = int(int_part_hex, 16) if int_part_hex else 0
    frac_value = int(frac_part_hex, 16) / 256.0  # 转换为0-255范围的小数

    return round(int_value + frac_value - 273.15, 1)


def parse_ip_port(ip_uint32, port_uint32):
    """
    将uint32格式的IP和端口转换为可读格式
    :param ip_uint32: 32位无符号整数表示的IP地址
    :param port_uint32: 32位无符号整数表示的端口
    :return: (ip_str, port) 元组
    :raises ValueError: 当端口超出0-65535范围时
    """
    # 验证端口范围
    if port_uint32 > 0xFFFF:
        logger.warning(f"Invalid port number: {port_uint32} (must be 0-65535)")
        port_uint32 = 12345

    # 解析IP地址
    ip_bytes = [
        (ip_uint32 >> 24) & 0xFF,
        (ip_uint32 >> 16) & 0xFF,
        (ip_uint32 >> 8) & 0xFF,
        ip_uint32 & 0xFF,
    ]
    ip_str = ".".join(map(str, ip_bytes))

    return ip_str, port_uint32


def get_bit_value(uint32_val, bit_id):
    """
    获取uint32数值中特定位的值（0或1）
    :param uint32_val: 32位无符号整数
    :param bit_id: 要查询的位序号（0-31，0表示最低位）
    :return: 0或1
    :raises ValueError: 当bit_id不在0-31范围内时
    """
    if not 0 <= bit_id <= 31:
        logger.warning("bit_id must be between 0 and 31,返回默认id=0")
        return (uint32_val >> 0) & 0x1
    return (uint32_val >> bit_id) & 0x1


def parse_bbox_from_uint32(xy_uint32, wh_uint32):
    """
    从组合的uint32值解析出bbox坐标[x,y,w,h]
    参数:
        xy_uint32: 组合的位置值（bit31-16:X中心, bit15-0:Y中心）
        wh_uint32: 组合的尺寸值（bit31-16:宽度W, bit15-0:高度H）
    返回:
        list: [x_left, y_top, w, h]
    """
    # 解析中心坐标
    x_center = (xy_uint32 >> 16) & 0xFFFF
    y_center = xy_uint32 & 0xFFFF

    # 解析宽高
    w = (wh_uint32 >> 16) & 0xFFFF
    h = wh_uint32 & 0xFFFF

    # # 转换为左上角坐标
    x_left = x_center - w // 2
    y_top = y_center - h // 2

    if x_left < 0:
        x_left = 0

    if y_top < 0:
        y_top = 0

    logger.debug(
        f"从组合的uint32值解析出bbox坐标： {x_center, y_center, w, h} --> {x_left, y_top, w, h}"
    )

    return [x_left, y_top, w, h]
    # return [x_center, y_center, w, h]


def map_boxes_to_crop_ltwh(ltwh_box, full_size=(2048, 2048), crop_size=640):
    """
    将全图 (left, top, width, height) 映射到中心 crop_size×crop_size 区域，
    并裁剪到 crop 边界内。

    Args:
        ltwh_box: (x_left, y_top, w, h) in 全图坐标
        full_size: (W_full, H_full)
        crop_size: 裁剪边长

    Returns:
        (x_left', y_top', w', h')，坐标均相对 crop 左上角(0,0)，
        且只返回与 crop 有交集的框，否则返回 []
    """
    W_full, H_full = full_size
    dx = (W_full - crop_size) // 2
    dy = (H_full - crop_size) // 2

    lx, ly, w, h = ltwh_box

    # 原框角点
    x1 = lx
    y1 = ly
    x2 = lx + w
    y2 = ly + h

    # 平移到 crop 坐标系
    nx1, ny1 = x1 - dx, y1 - dy
    nx2, ny2 = x2 - dx, y2 - dy

    # 剔除完全在 crop 之外的框
    if nx2 <= 0 or ny2 <= 0 or nx1 >= crop_size or ny1 >= crop_size:
        return []

    # 截断到 crop 内
    nx1 = max(0, min(crop_size, nx1))
    ny1 = max(0, min(crop_size, ny1))
    nx2 = max(0, min(crop_size, nx2))
    ny2 = max(0, min(crop_size, ny2))

    # 回到 ltwh
    new_w = nx2 - nx1
    new_h = ny2 - ny1

    if new_w > 0 and new_h > 0:
        return [nx1, ny1, new_w, new_h]

    return []
