# calibrator.py
import numpy as np
from scipy.interpolate import interp1d

__focus2pot_interp = None
__pot2focus_interp = None

poly_fp = None
poly_pf = None


def init_calib(npz_path: str, kind: str = "cubic", fill_value="extrapolate") -> None:
    """
    读取 npz 校准文件，构建两个插值器：
      1) focallength → pot
      2) pot   → fofocallengthus
    系统启动时调用一次。
    """
    global poly_fp, poly_pf

    data = np.load(npz_path)
    focal_lens = data["focal_lens"]
    pot_vals = data["pot_vals"]

    # focallength -> pot
    coef_fp = np.polyfit(focal_lens, pot_vals, 8)
    poly_fp = np.poly1d(coef_fp)

    coef_pf = np.polyfit(pot_vals, focal_lens, 8)
    poly_pf = np.poly1d(coef_pf)


def focallength_to_pot(focallength: float | np.ndarray) -> float | np.ndarray:
    """
    给定焦距（或焦距数组），返回对应电位器值。
    必须先调用 init_calib()。
    """

    if poly_fp is None:
        return RuntimeError("请先调用 init_calib() 初始化校准数据")
    return poly_fp(focallength)


def pot_to_focallength(pot: float | np.ndarray) -> float | np.ndarray:
    """
    给定电位器值（或数组），返回对应焦距（mm）。
    必须先调用 init_calib()。
    """

    if poly_pf is None:
        return RuntimeError("请先调用 init_calib() 初始化校准数据")
    return poly_pf(pot)
