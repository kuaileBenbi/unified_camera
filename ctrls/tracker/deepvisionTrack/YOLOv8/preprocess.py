import cv2
import numpy as np


def letterbox(
    im: np.ndarray,
    new_shape=(640, 640),
    color=(0, 0, 0),
    scaleup: bool = True,
    reuse_out: np.ndarray = None,
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    """
    Resize and pad image to `new_shape` (h, w) with minimal allocations.
    Returns: (out, (r_w, r_h), (dw, dh))
      - out:      resized+padded image
      - ratio:    (width_ratio, height_ratio)
      - padding:  (dw, dh) in float (total padding per axis, before split)
    Args:
      im:         H×W×C uint8 input
      new_shape:  target (height, width)
      color:      padding color
      scaleup:    allow upscaling (False will only downscale)
      reuse_out:  optional buffer of shape (new_h, new_w, C) to reuse
    """
    orig_h, orig_w = im.shape[:2]
    target_h, target_w = new_shape

    # 1. 计算缩放比例
    r = min(target_h / orig_h, target_w / orig_w)
    if not scaleup:
        r = min(r, 1.0)
    new_unpad_w, new_unpad_h = int(round(orig_w * r)), int(round(orig_h * r))
    dw, dh = target_w - new_unpad_w, target_h - new_unpad_h
    dw /= 2  # padding on left/right
    dh /= 2  # padding on top/bottom

    # 2. 预分配输出画布（可复用）
    if reuse_out is None or reuse_out.shape[:2] != (target_h, target_w):
        if im.ndim == 3:
            out = np.full((target_h, target_w, im.shape[2]), color, dtype=im.dtype)
        else:
            out = np.full((target_h, target_w), color[0], dtype=im.dtype)
    else:
        out = reuse_out
        # 先填充背景色
        out[:] = color

    # 3. 缩放并写入输出
    if (orig_w, orig_h) != (new_unpad_w, new_unpad_h):
        resized = cv2.resize(
            im, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR
        )
    else:
        resized = im

    # 4. 计算放置位置并原地赋值
    top = int(round(dh))
    left = int(round(dw))
    out[top : top + new_unpad_h, left : left + new_unpad_w] = resized

    return out, (r, r), (dw, dh)
