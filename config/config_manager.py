import os
import toml
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path


class ConfigManager:
    """
    统一配置管理器
    管理所有相机模式的配置文件
    """
    
    def __init__(self, config_base_path: str = "configs"):
        """
        初始化配置管理器
        
        Args:
            config_base_path: 配置文件基础路径
        """
        self.config_base_path = Path(config_base_path)
        self._configs = {}
        self._load_all_configs()
    
    def _load_all_configs(self):
        """加载所有模式的配置文件"""
        modes = ["lwir_fix", "mwir_fix", "mwir_zoom", "swir_fix", "vis_fix", "vis_zoom"]
        
        for mode in modes:
            try:
                # 加载相机配置
                cam_config_path = self.config_base_path / mode / "cam_config.toml"
                if cam_config_path.exists():
                    with open(cam_config_path, 'r', encoding='utf-8') as f:
                        cam_config = toml.load(f)
                    self._configs[f"{mode}_camera"] = cam_config
                
                # 加载图像处理配置
                imager_config_path = self.config_base_path / mode / "pipelines.yaml"
                if imager_config_path.exists():
                    with open(imager_config_path, 'r', encoding='utf-8') as f:
                        imager_config = yaml.safe_load(f)
                    self._configs[f"{mode}_imager"] = imager_config
                    
            except Exception as e:
                print(f"加载 {mode} 配置失败: {e}")
    
    def get_camera_config(self, mode: str) -> Dict[str, Any]:
        """
        获取指定模式的相机配置
        
        Args:
            mode: 相机模式
            
        Returns:
            相机配置字典
        """
        config_key = f"{mode}_camera"
        if config_key not in self._configs:
            raise ValueError(f"未找到模式 {mode} 的相机配置")
        return self._configs[config_key]
    
    def get_imager_config(self, mode: str) -> Dict[str, Any]:
        """
        获取指定模式的图像处理配置
        
        Args:
            mode: 相机模式
            
        Returns:
            图像处理配置字典
        """
        config_key = f"{mode}_imager"
        if config_key not in self._configs:
            raise ValueError(f"未找到模式 {mode} 的图像处理配置")
        return self._configs[config_key]
    
    def get_all_modes(self) -> List[str]:
        """获取所有可用的相机模式"""
        modes = []
        for key in self._configs.keys():
            if key.endswith("_camera"):
                mode = key.replace("_camera", "")
                modes.append(mode)
        return modes
    
    def update_config(self, mode: str, config_type: str, updates: Dict[str, Any]):
        """
        更新配置
        
        Args:
            mode: 相机模式
            config_type: 配置类型 (camera 或 imager)
            updates: 要更新的配置项
        """
        config_key = f"{mode}_{config_type}"
        if config_key in self._configs:
            self._configs[config_key].update(updates)
        else:
            raise ValueError(f"未找到配置 {config_key}")
    
    def save_config(self, mode: str, config_type: str):
        """
        保存配置到文件
        
        Args:
            mode: 相机模式
            config_type: 配置类型 (camera 或 imager)
        """
        config_key = f"{mode}_{config_type}"
        if config_key not in self._configs:
            raise ValueError(f"未找到配置 {config_key}")
        
        config_data = self._configs[config_key]
        
        if config_type == "camera":
            config_path = self.config_base_path / mode / "cam_config.toml"
            with open(config_path, 'w', encoding='utf-8') as f:
                toml.dump(config_data, f)
        elif config_type == "imager":
            config_path = self.config_base_path / mode / "pipelines.yaml"
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
