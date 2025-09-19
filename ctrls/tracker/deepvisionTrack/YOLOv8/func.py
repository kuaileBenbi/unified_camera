import cv2
import numpy as np

from .preprocess import letterbox
from .postprocess import yolov8_postprocess


def myFunc(rknn_lite, IMG):
    # 等比例缩放
    _IMG, ratio, (dw, dh) = letterbox(IMG)

    IMG2 = np.expand_dims(_IMG, 0)

    outputs = rknn_lite.inference(inputs=[IMG2], data_format=["nhwc"])

    ltrb_boxes, classes_id, scores = yolov8_postprocess(
        outputs, IMG.shape, ratio, dw, dh
    )

    outputs = None
    del outputs
    IMG2 = None
    del IMG2

    return IMG, {
        "ltrb_boxes": ltrb_boxes,
        "classes_id": classes_id,
        "scores": scores,
    }
