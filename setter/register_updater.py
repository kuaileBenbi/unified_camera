"""
register_updater module

Provides update_registers() to update camera-specific registers based on state values.
"""

# try:
#     from status import set_values
# except Exception:
import os, sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from status import set_values
import logging

# 默认日志记录器，会被具体的相机模式日志记录器替换
logger = logging.getLogger(__name__)


# 全局日志记录器设置函数
def set_register_updater_logger(camera_logger):
    """设置register_updater模块的日志记录器"""
    global logger
    logger = camera_logger


# Mapping of camera types to their register keys
REGISTER_MAP = {
    "vis_zoom": [
        "vis_zoom_hfov",
        "vis_zoom_integ_time_setting",
        "vis_zoom_gain_setting",
        "vis_zoom_focus_abs_position",
        "vis_zoom_sharpness_evaluation",
    ],
    "mwir_zoom": [
        "mwir_zoom_hfov",
        "mwir_zoom_status1",
        "mwir_zoom_status2",
        "mwir_zoom_integ_time",
        "mwir_zoom_gain",
        "mwir_zoom_temperature",
        "mwir_zoom_brightness_avg",
        "mwir_zoom_auto_brightness_status",
        "mwir_zoom_elec_zoom_status",
        "mwir_zoom_elec_zoom_setting",
    ],
    "vis_fix": [
        "vis_fix_integ_time_setting",
        "vis_fix_gain_setting",
        "vis_fix_focus_abs_position",
        "vis_fix_sharpness_evaluation",
    ],
    "swir_fix": [
        "swir_fix_integ_time_setting",
        "swir_fix_gain_setting",
        "swir_fix_focus_abs_position",
        "swir_fix_sharpness_evaluation",
        "swir_fix_camera_temperature",
        "swir_fix_camera_tec_current",
    ],
    "mwir_fix": [
        "mwir_fix_integ_time_setting",
        "mwir_fix_gain_setting",
        "mwir_fix_focus_abs_position",
        "mwir_fix_sharpness_evaluation",
        "mwir_fix_camera_temperature",
    ],
    "lwir_fix": [
        "lwir_fix_integ_time_setting",
        "lwir_fix_gain_setting",
        "lwir_fix_focus_abs_position",
        "lwir_fix_sharpness_evaluation",
        "lwir_fix_camera_temperature",
    ],
}


def update_registers(state: dict, camera_type: str) -> bool:
    """
    Update register values for the given camera type based on provided state dict.

    :param state: A dict mapping register suffixes to their new values, e.g. {'hfov': 45, 'status': 1, ...}.
    :param camera_type: One of the keys in REGISTER_MAP (e.g., 'vis_zoom', 'mwir_fix').
    """
    if not state:
        logger.warning("No state provided, skipping register update.")
        return False
    if camera_type not in REGISTER_MAP:
        logger.error(f"Unsupported camera type: {camera_type}")
        return False

    full_keys = REGISTER_MAP[camera_type]
    status = {}

    prefix = camera_type + "_"  # vis_zoom_
    for full_key in full_keys:
        # derive suffix by removing prefix
        if full_key.startswith(prefix):
            suffix = full_key[len(prefix) :]
        else:
            suffix = full_key.split("_", 1)[-1]

        if suffix in state:
            status[full_key] = state[suffix]

    # Call the external function to update registers
    if status:
        try:
            set_values(status)
            # logger.debug(f"更新{camera_type}状态: {status}")
        except Exception as e:
            logger.error(f"Failed to update registers: {e}")

    return True


# Allow running as script for quick tests
if __name__ == "__main__":
    example_state = {
        "hfov": 45,
        "status": 1,
        "integ_time_setting": 1000,
        "gain_setting": 10,
        "focus_abs_position": 500,
        "sharpness_evaluation": 200,
    }
    update_registers(example_state, "vis_zoom")
    print("Registers updated successfully.")
