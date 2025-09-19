from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import cv2

from .utils.bbox import corner2center

from rknnlite.api import RKNNLite


class NnoTracker:
    def __init__(self, cfg, Tback_weight, Xback_weight, Head_weight):

        self.cfg = cfg

        self.score_size = (
            (self.cfg.TRACK.INSTANCE_SIZE - self.cfg.TRACK.EXEMPLAR_SIZE)
            // self.cfg.POINT.STRIDE
            + 1
            + self.cfg.TRACK.BASE_SIZE
        )
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.cls_out_channels = 2
        self.window = window.flatten()

        self.points = self.generate_points(self.cfg.POINT.STRIDE, self.score_size)

        # --------------------------------------------------------#
        # --------------modify environment------------------------#
        # 1. T init
        self.rknn_Tback = RKNNLite()

        # load RKNN model
        print("--> Load RKNN model")
        ret = self.rknn_Tback.load_rknn(Tback_weight)
        if ret != 0:
            print("Load RKNN model failed")
            exit(ret)
        print("done")

        # init runtime environment
        print("--> Init runtime environment")

        ret = self.rknn_Tback.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        if ret != 0:
            print("Init runtime environment failed")
            exit(ret)
        print("done")

        # 2. X init
        self.rknn_Xback = RKNNLite()

        # Load model
        print("--> rknn_Xback: Loading model")
        ret = self.rknn_Xback.load_rknn(Xback_weight)
        if ret != 0:
            print("rknn_Xback: Load model failed!")
            exit(ret)
        print("rknn_Xback:done")

        # Init runtime environment
        print("--> Init runtime environment")
        ret = self.rknn_Xback.init_runtime(core_mask=RKNNLite.NPU_CORE_1)
        if ret != 0:
            print("Init runtime environment failed!")
            exit(ret)
        print("done")

        # 3. Head init
        self.rknn_Head = RKNNLite()

        # Load model
        print("--> rknn_Head: Loading model")
        ret = self.rknn_Head.load_rknn(Head_weight)
        if ret != 0:
            print("rknn_Head: Load model failed!")
            exit(ret)
        print("rknn_Head:done")

        # Init runtime environment
        print("--> Init runtime environment")
        ret = self.rknn_Head.init_runtime(core_mask=RKNNLite.NPU_CORE_2)
        if ret != 0:
            print("Init runtime environment failed!")
            exit(ret)
        print("done")

    def generate_points(self, stride, size):
        ori = -(size // 2) * stride
        x, y = np.meshgrid(
            [ori + stride * dx for dx in np.arange(0, size)],
            [ori + stride * dy for dy in np.arange(0, size)],
        )
        points = np.zeros((size * size, 2), dtype=np.float32)
        points[:, 0], points[:, 1] = (
            x.astype(np.float32).flatten(),
            y.astype(np.float32).flatten(),
        )

        return points

    def _convert_bbox(self, delta, point):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.detach().cpu().numpy()

        delta[0, :] = point[:, 0] - delta[0, :]  # x1
        delta[1, :] = point[:, 1] - delta[1, :]  # y1
        delta[2, :] = point[:, 0] + delta[2, :]  # x2
        delta[3, :] = point[:, 1] + delta[3, :]  # y2
        delta[0, :], delta[1, :], delta[2, :], delta[3, :] = corner2center(delta)
        return delta

    def _convert_score(self, score):
        if self.cls_out_channels == 1:
            score = score.permute(1, 2, 3, 0).contiguous().view(-1)
            score = score.sigmoid().detach().cpu().numpy()
        else:
            score = (
                score.permute(1, 2, 3, 0)
                .contiguous()
                .view(self.cls_out_channels, -1)
                .permute(1, 0)
            )
            score = score.softmax(1).detach()[:, 1].cpu().numpy()
        return score

    def _convert_bbox_numpy(self, delta, point):
        # delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        # delta = delta.detach().cpu().numpy()

        delta = delta.transpose((1, 2, 3, 0)).reshape(4, -1)

        delta[0, :] = point[:, 0] - delta[0, :]  # x1
        delta[1, :] = point[:, 1] - delta[1, :]  # y1
        delta[2, :] = point[:, 0] + delta[2, :]  # x2
        delta[3, :] = point[:, 1] + delta[3, :]  # y2
        delta[0, :], delta[1, :], delta[2, :], delta[3, :] = corner2center(delta)
        return delta

    def _convert_score_numpy(self, score):
        def sofmax(logits):
            e_x = np.exp(logits)
            probs = e_x / np.sum(e_x, axis=-1, keepdims=True)
            return probs

        # score = score.permute(1, 2, 3, 0).contiguous().view(self.cls_out_channels, -1).permute(1, 0)
        # score = score.softmax(1).detach()[:, 1].cpu().numpy()

        score = (
            score.transpose((1, 2, 3, 0))
            .reshape(self.cls_out_channels, -1)
            .transpose((1, 0))
        )
        score = sofmax(score)[:, 1]

        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def get_subwindow(self, im, pos, model_sz, original_sz, avg_chans):
        """
        args:
            im: bgr based image
            pos: center position
            model_sz: exemplar size
            s_z: original size
            avg_chans: channel average
        """
        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        # context_xmin = round(pos[0] - c) # py2 and py3 round
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        # context_ymin = round(pos[1] - c)
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0.0, -context_xmin))
        top_pad = int(max(0.0, -context_ymin))
        right_pad = int(max(0.0, context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0.0, context_ymax - im_sz[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        r, c, k = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
            te_im = np.zeros(size, np.uint8)
            te_im[top_pad : top_pad + r, left_pad : left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad : left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad :, left_pad : left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c + left_pad :, :] = avg_chans
            im_patch = te_im[
                int(context_ymin) : int(context_ymax + 1),
                int(context_xmin) : int(context_xmax + 1),
                :,
            ]
        else:
            im_patch = im[
                int(context_ymin) : int(context_ymax + 1),
                int(context_xmin) : int(context_xmax + 1),
                :,
            ]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch, (model_sz, model_sz))
        im_patch = im_patch.transpose(2, 0, 1)
        im_patch = im_patch[np.newaxis, :, :, :]
        im_patch = im_patch.astype(np.float32)

        return im_patch

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.center_pos = np.array(
            [bbox[0] + (bbox[2] - 1) / 2, bbox[1] + (bbox[3] - 1) / 2]
        )
        # 1. 原始目标框的宽高
        self.size = np.array([bbox[2], bbox[3]])

        # 2. 计算包含上下文的裁剪区域宽高
        w_z = self.size[0] + self.cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + self.cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        # 3. 取两者的几何平均得到最终正方形边长
        s_z = round(np.sqrt(w_z * h_z))

        # 5. channel_average 用于边界填充
        self.channel_average = np.mean(img, axis=(0, 1))

        # 6. 裁剪并缩放
        z_crop = self.get_subwindow(
            img,
            self.center_pos,
            self.cfg.TRACK.EXEMPLAR_SIZE,  # 4. 网络输入尺寸
            s_z,
            self.channel_average,
        )

        back_T_in = z_crop.transpose((0, 2, 3, 1))

        try:
            Toutput = self.rknn_Tback.inference(inputs=[back_T_in])
            self.Toutput = Toutput

            Toutput = None
            del Toutput

        except Exception as e:
            print(f"提取模板遇到错误：{e}")
            return False

        finally:
            self.rknn_Tback.release()

        return True

    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height] -> 修改为xyxy: ltrb_boxes
        """

        # 1. 延续上一次 track/ init 后保存的 self.size = [W, H]
        w_z = self.size[0] + self.cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + self.cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        # 2. 计算模板到搜索区域的缩放比例
        scale_z = self.cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (self.cfg.TRACK.INSTANCE_SIZE / self.cfg.TRACK.EXEMPLAR_SIZE)
        # 4. 裁剪并缩放
        x_crop = self.get_subwindow(
            img,
            self.center_pos,  # 仍然以上一帧预测的中心为中心
            self.cfg.TRACK.INSTANCE_SIZE,  # 3. 搜索区域在原图上的大小
            round(s_x),  # 在原图上截取 s_x×s_x 大小的区域
            self.channel_average,
        )

        # predict
        back_X_in = x_crop.transpose((0, 2, 3, 1))
        Xoutput = self.rknn_Xback.inference(inputs=[back_X_in])
        self.Xoutput = Xoutput

        Xoutput = None
        del Xoutput

        head_T_in = self.Toutput[0].transpose((0, 2, 3, 1))
        head_X_in = self.Xoutput[0].transpose((0, 2, 3, 1))

        _outputs = self.rknn_Head.inference(inputs=[head_T_in, head_X_in])

        outputs = _outputs

        _outputs = None
        del _outputs

        score = self._convert_score_numpy(outputs[0])
        pred_bbox = self._convert_bbox_numpy(outputs[1], self.points)

        def change(r, eps=1e-8):
            return np.maximum(r, 1.0 / (r + eps))

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(
            sz(pred_bbox[2, :], pred_bbox[3, :])
            / (sz(self.size[0] * scale_z, self.size[1] * scale_z))
        )

        # aspect ratio penalty
        r_c = change(
            (self.size[0] / self.size[1]) / (pred_bbox[2, :] / pred_bbox[3, :])
        )
        penalty = np.exp(-(r_c * s_c - 1) * self.cfg.TRACK.PENALTY_K)

        # score
        pscore = penalty * score

        # window penalty
        pscore = (
            pscore * (1 - self.cfg.TRACK.WINDOW_INFLUENCE)
            + self.window * self.cfg.TRACK.WINDOW_INFLUENCE
        )

        best_idx = np.argmax(pscore)

        best_score = score[best_idx]

        if best_score < self.cfg.TRACK.MIN_SCORE:
            return {"bbox": [], "best_score": best_score}

        bbox = pred_bbox[:, best_idx] / scale_z

        lr = penalty[best_idx] * score[best_idx] * self.cfg.TRACK.LR

        cx = bbox[0] + self.center_pos[0]

        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr

        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width, height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        # bbox = [cx - width / 2, cy - height / 2, width, height]
        bbox = [cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2]

        return {"bbox": bbox, "best_score": best_score}

    def _release(self):

        if hasattr(self, "rknn_Tback") and self.rknn_Tback is not None:
            try:
                self.rknn_Tback.release()
            except Exception as e:
                print(f"[WARN] T_RKNN release failed: {e}")
                return False

        if hasattr(self, "rknn_Xback") and self.rknn_Xback is not None:
            try:
                self.rknn_Xback.release()
            except Exception as e:
                print(f"[WARN] X_RKNN release failed: {e}")
                return False

        if hasattr(self, "rknn_Head") and self.rknn_Head is not None:
            try:
                self.rknn_Head.release()
            except Exception as e:
                print(f"[WARN] HEAD_RKNN release failed: {e}")
                return False

        return True
