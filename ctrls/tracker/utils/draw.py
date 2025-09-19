import numpy as np
import cv2

palette = (2**11 - 1, 2**15 - 1, 2**20 - 1)

# 定义类别与位的映射
CLASS_COLOR_MAPPING = {
    "boat": (255, 182, 193),  # Light pink for "boat" (RGB format)
    "aeroplane": (255, 250, 205),  # Light yellow for "aeroplane" (RGB format)
    "car": (144, 238, 144),  # Light green for "car" (RGB format)
    "person": (135, 206, 235),
}

font = cv2.FONT_HERSHEY_TRIPLEX
font_scale = 1.0
thickness = 2
fallback_color = (0, 255, 0)


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label**2 - label + 1)) % 255) for p in palette]
    return tuple(color)


# def draw_boxes(img, bbox, names=None, identities=None):
#     color = [255, 0, 0]

#     for i, box in enumerate(bbox[:5]):
#         # print(i)
#         x1, y1, x2, y2 = [int(d) for d in box]

#         if identities is not None:
#             # print(f"identities: {identities[i]}")
#             tar_id = int(identities[i])
#             color = compute_color_for_labels(tar_id)
#             label = f"label:{names[i]} id:{tar_id}"

#         else:
#             # print(f"identities: {identities}")
#             label = f"{names[i]}"

#         cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
#         # width = abs(x2 - x1)
#         # height = abs(y2 - y1)
#         # if width > 255 or height > 255:
#         #     # 如果满足条件，保存图像到本地
#         #     cv2.imwrite(f"large_rectangle_image_{i}.jpg", img)
#         cv2.putText(
#             img, label, (x1, y1 - 6), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 1
#         )
#     return img


# def draw(image, boxes, classes, ids):

#     for bbox, cl_name, t_id in zip(boxes, classes, ids):
#         l, t, r, b = (int(x) for x in bbox)

#         cv2.rectangle(image, (l, t), (r, b), (255, 0, 0), 2)
#         cv2.putText(
#             image,
#             f"{cl_name} id: {t_id}",
#             (l, t - 6),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.6,
#             (0, 0, 255),
#             2,
#         )


# def draw_boxes(img, bbox, names=None, identities=None):
#     for i, box in enumerate(bbox[:4]):
#         x1, y1, x2, y2 = [int(d) for d in box]

#         # if identities is not None:
#         #     print(identities[i])

#         # Scaling the box if width or height is greater than 255
#         width = x2 - x1
#         height = y2 - y1

#         if width > 255 or height > 255:
#             # Scale to 255x255
#             if width > height:
#                 scale_factor = 255.0 / width
#             else:
#                 scale_factor = 255.0 / height

#             # Calculate the new dimensions while keeping the center of the box
#             center_x = (x1 + x2) // 2
#             center_y = (y1 + y2) // 2
#             new_width = int(width * scale_factor)
#             new_height = int(height * scale_factor)

#             # Recalculate the new x1, y1, x2, y2 to keep the box centered
#             x1 = max(center_x - new_width // 2, 0)
#             y1 = max(center_y - new_height // 2, 0)
#             x2 = min(center_x + new_width // 2, img.shape[1])
#             y2 = min(center_y + new_height // 2, img.shape[0])

#         # Color depends on identity (if available) or uses a default color

#         # label = (
#         #     f"label:{names[i]} id:{int(identities[i])}"
#         #     if identities is not None
#         #     else f"{names[i]}"
#         # )
#         label = f"{names[i]} id:{i}"
#         # print(f"置信度：{identities[i]}  id:{i}")

#         # color = (
#         #     compute_color_for_labels(int(identities[i]))
#         #     if identities is not None
#         #     else (0, 255, 0)
#         # )

#         color = CLASS_COLOR_MAPPING[names[i]] if identities is not None else (0, 255, 0)

#         thickness = 2
#         cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

#         label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 2)
#         label_x = x1
#         label_y = y1 - label_size[1]
#         cv2.rectangle(
#             img,
#             (label_x, label_y),
#             (label_x + label_size[0], label_y + label_size[1]),
#             color,
#             cv2.FILLED,
#         )
#         cv2.putText(
#             img,
#             label,
#             (label_x, label_y + label_size[1] - 3),
#             cv2.FONT_HERSHEY_PLAIN,
#             1,
#             [255, 255, 255],
#             2,
#         )

#     return img


def check_wh(x1, y1, x2, y2, img_w, img_h):
    width = x2 - x1
    height = y2 - y1

    if width > 255 or height > 255:
        # Scale to 255x255
        if width > height:
            scale_factor = 255.0 / width
        else:
            scale_factor = 255.0 / height

        # Calculate the new dimensions while keeping the center of the box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        # Recalculate the new x1, y1, x2, y2 to keep the box centered
        x1 = max(center_x - new_width // 2, 0)
        y1 = max(center_y - new_height // 2, 0)
        x2 = min(center_x + new_width // 2, img_w)  # img.shape[1]
        y2 = min(center_y + new_height // 2, img_h)  # img.shape[0]

    return x1, y1, x2, y2


def check_color(names, i):
    if names is not None:
        cls_name = names[i]
        if cls_name in CLASS_COLOR_MAPPING:
            color = CLASS_COLOR_MAPPING[cls_name]
        else:
            color = fallback_color
            print(
                f"Warning: class name '{cls_name}' not found in CLASS_COLOR_MAPPING. Using fallback color {fallback_color}."
            )
    else:
        color = fallback_color

    return color


def put_label(
    im,
    x1,
    y1,
    x2,
    y2,
    label,
    flip_mode="none",
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.6,
    text_thickness=1,
    box_thickness=2,
    color=(0, 255, 0),
):
    H, W = im.shape[:2]
    (tw, th), base = cv2.getTextSize(label, font, font_scale, text_thickness)
    pad = 2

    # 依据方向选择标签放置侧
    # 原图/水平/垂直翻转 -> 仍放在框“上方”
    # rot90ccw -> 放“左侧”；rot90cw -> 放“右侧”；rot180 -> 放“下方”
    mode = flip_mode.lower() if flip_mode else "none"
    if mode in ("none", "flip_h", "flip_v", "flip_hv"):
        tx1, ty2 = x1, max(th + base + pad, y1)  # 上侧
        tx2, ty1 = min(W - 1, x1 + tw + 2 * pad), ty2 - (th + base + 2 * pad)
    elif mode == "rot90ccw":
        tx1 = max(0, x1 - tw - 2 * pad)  # 左侧
        tx2 = x1 - 1
        ty1 = max(0, y1)
        ty2 = min(H - 1, y1 + th + 2 * pad)
    elif mode == "rot90cw":
        tx1 = min(W - 1, x2 + 1)  # 右侧
        tx2 = min(W - 1, x2 + tw + 2 * pad)
        ty1 = max(0, y1)
        ty2 = min(H - 1, y1 + th + 2 * pad)
    elif mode == "rot180":
        tx1 = x1  # 下侧
        ty1 = min(H - 1, y2 + base + pad)
        ty2 = min(H - 1, ty1 + th + base + pad)
        tx2 = min(W - 1, x1 + tw + 2 * pad)
    else:
        # 未知模式，退化为上侧
        tx1, ty2 = x1, max(th + base + pad, y1)
        tx2, ty1 = min(W - 1, x1 + tw + 2 * pad), ty2 - (th + base + 2 * pad)

    # 再次裁剪
    tx1 = max(0, min(W - 1, tx1))
    tx2 = max(0, min(W - 1, tx2))
    ty1 = max(0, min(H - 1, ty1))
    ty2 = max(0, min(H - 1, ty2))

    # 背景块
    cv2.rectangle(im, (tx1, ty1), (tx2, ty2), color, thickness=cv2.FILLED)

    # 文本基线点（不同侧要调）
    if mode in ("none", "flip_h", "flip_v", "flip_hv"):
        org = (tx1 + pad, ty2 - base - pad)
    elif mode == "rot90ccw":
        org = (tx1 + pad, min(ty1 + th + pad, H - 1))
    elif mode == "rot90cw":
        org = (tx1 + pad, min(ty1 + th + pad, H - 1))
    elif mode == "rot180":
        org = (tx1 + pad, min(ty2 - base - pad, H - 1))
    else:
        org = (tx1 + pad, ty2 - base - pad)

    cv2.putText(
        im, label, org, font, font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA
    )


def draw_boxes(
    img,
    bbox,
    names=None,
    identities=None,
    tracking=False,
    flip_mode="none",
):
    im = np.ascontiguousarray(np.copy(img))

    for i, box in enumerate(bbox):
        if not tracking:
            x1, y1, x2, y2 = [int(x) for x in box]
        else:
            x, y, w, h = box
            x1, y1, x2, y2 = map(int, (x, y, x + w, y + h))

        x1, y1, x2, y2 = check_wh(x1, y1, x2, y2, im.shape[1], im.shape[0])
        obj_id = int(identities[i]) if identities is not None else i

        color = compute_color_for_labels(obj_id)

        label = "{:} {:d}".format(names[i], obj_id)

        # 依据方向选择标签放置侧
        # 原图/水平/垂直翻转 -> 仍放在框“上方”
        # rot90ccw -> 放“左侧”；

        cv2.rectangle(im, (x1, y1), (x2, y2), color=color, thickness=2)

        put_label(
            im,
            x1,
            y1,
            x2,
            y2,
            label,
            flip_mode=flip_mode,
            color=color,
            font=font,
            font_scale=font_scale,
            text_thickness=thickness,
        )

    return im
