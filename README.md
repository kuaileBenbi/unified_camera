# 统一相机控制系统

一个支持多种相机模式的统一相机控制和管理系统，专为光学设备集成设计。

## 📋 项目概述

本项目是一个统一的相机控制框架，支持多种类型的相机设备，包括可见光、红外（短波、中波、长波）等不同波段的相机，以及固定焦距和变焦相机。系统提供了完整的相机控制、图像处理、目标跟踪和网络通信功能。

## 🎯 支持的相机模式

| 模式          | 描述                 | 焦距类型 |
| ------------- | -------------------- | -------- |
| `lwir_fix`  | 长波红外固定焦距相机 | 固定     |
| `mwir_fix`  | 中波红外固定焦距相机 | 固定     |
| `mwir_zoom` | 中波红外变焦相机     | 变焦     |
| `swir_fix`  | 短波红外固定焦距相机 | 固定     |
| `vis_fix`   | 可见光固定焦距相机   | 固定     |
| `vis_zoom`  | 可见光变焦相机       | 变焦     |

## 🚀 主要功能

### 核心功能

- **统一相机控制**: 支持多种相机类型的统一控制接口
- **实时图像处理**: 基于GStreamer的高性能图像处理管道
- **目标检测与跟踪**: 集成YOLOv8和模板匹配跟踪算法
- **网络通信**: 支持ZMQ消息总线和UDP视频流传输
- **配置管理**: 灵活的TOML/YAML配置文件系统
- **异步处理**: 基于asyncio的高并发处理架构

### 技术特性

- **多线程架构**: 图像采集、处理、控制分离的并发处理
- **硬件加速**: 支持RKNN神经网络推理加速
- **实时控制**: V4L2相机参数实时调节
- **状态管理**: 完整的设备状态监控和管理
- **日志系统**: 分级日志记录和调试支持

## 📁 项目结构

```
unified_camera/
├── core/                    # 核心控制模块
│   └── unified_camera.py   # 统一相机控制器
├── config/                  # 配置管理
│   └── config_manager.py   # 配置管理器
├── configs/                 # 配置文件目录
│   ├── lwir_fix/           # 长波红外固定焦距配置
│   ├── mwir_fix/           # 中波红外固定焦距配置
│   ├── mwir_zoom/          # 中波红外变焦配置
│   ├── swir_fix/           # 短波红外固定焦距配置
│   ├── vis_fix/            # 可见光固定焦距配置
│   └── vis_zoom/           # 可见光变焦配置
├── ctrls/                   # 相机控制模块
│   ├── calibrator.py       # 校准器
│   ├── cam_with_pre.py     # 带预处理相机控制
│   ├── cap_with_pre.py     # 带预处理采集控制
│   ├── cap_no_pre_vis.py   # 无预处理可见光采集
│   ├── v4l2ctrlor.py       # V4L2控制接口
│   └── tracker/            # 目标跟踪模块
│       ├── deepvisionTrack/ # 深度学习跟踪
│       └── templaterTrack/  # 模板匹配跟踪
├── imager/                  # 图像处理模块
│   ├── gstworker.py        # GStreamer工作器
│   ├── preworker.py        # 预处理工作器
│   ├── detworker.py        # 检测工作器
│   └── updater.py          # 状态更新器
├── bus/                     # 消息总线
│   └── bus.py              # 事件总线实现
├── setter/                  # 寄存器设置
│   └── register_updater.py  # 寄存器更新器
├── utils/                   # 工具模块
│   ├── logger.py           # 日志工具
│   └── utils.py            # 通用工具
└── entry.py                # 程序入口
```

## 🛠️ 安装和配置

### 系统要求

- Python 3.8+
- Linux系统（推荐Ubuntu 20.04+）
- GStreamer 1.16+
- V4L2支持
- RKNN运行时（可选，用于AI推理加速）

### 安装步骤

1. **克隆项目**

   ```bash
   git clone <repository-url>
   cd unified_camera
   ```
2. **安装依赖**

   ```bash
   pip install -r requirements.txt
   ```
3. **安装系统依赖**

   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install python3-gi python3-gi-cairo gir1.2-gstreamer-1.0 \
                        gir1.2-gst-plugins-base-1.0 gstreamer1.0-plugins-* \
                        v4l-utils libv4l-dev
   ```
4. **配置相机设备**

   ```bash
   # 检查可用的视频设备
   v4l2-ctl --list-devices

   # 设置设备权限
   sudo chmod 666 /dev/video*
   ```

### 配置文件

每个相机模式都有独立的配置目录，包含：

- `cam_config.toml`: 相机参数配置
- `pipelines.yaml`: GStreamer处理管道配置
- `pipelines_test.yaml`: 测试管道配置

## 🎮 使用方法

### 基本使用

```bash
# 启动可见光固定焦距相机
python entry.py --mode vis_fix

# 启动中波红外变焦相机
python entry.py --mode mwir_zoom

# 指定配置路径和日志级别
python entry.py --mode lwir_fix --config-path ./configs --log-level DEBUG
```

### 命令行参数

| 参数              | 描述             | 默认值      |
| ----------------- | ---------------- | ----------- |
| `--mode`        | 相机模式（必需） | 无          |
| `--config-path` | 配置文件路径     | `configs` |
| `--log-level`   | 日志级别         | `INFO`    |
| `--log-dir`     | 日志文件目录     | `logs`    |
| `--version`     | 显示版本信息     | -           |

### 支持的日志级别

- `DEBUG`: 详细调试信息
- `INFO`: 一般信息
- `WARNING`: 警告信息
- `ERROR`: 错误信息
- `CRITICAL`: 严重错误

## 🔧 配置说明

### 相机配置 (cam_config.toml)

```toml
[setting]
title = "vis_fix"
cam_entry = "192.168.137.102:5599"  # 相机控制端口
cap_entry = "192.168.137.102:5577"   # 图像采集端口
cameraid = 2                          # 相机ID
image_width = 2048                    # 图像宽度
image_height = 2048                   # 图像高度
camera_power = 1                      # 相机电源控制

[setting.encoder_params]
fps = 30                             # 帧率
bps = 5000000                        # 比特率
udp_ip = "10.16.22.210"             # UDP目标IP
udp_port = 8210                      # UDP目标端口
```

### 处理管道配置 (pipelines.yaml)

```yaml
pipelines:
  live_encoder:
    type: gst
    template: >
      v4l2src device=$device io-mode=2 !
      video/x-raw,framerate=25/1 !
      videoconvert !
      video/x-raw,format=GRAY8 !
      # ... 更多处理步骤
```

## 🎯 目标跟踪

系统集成了两种跟踪算法：

### 深度学习跟踪 (DeepVisionTrack)

- 基于YOLOv8的目标检测
- BYTETracker多目标跟踪
- RKNN硬件加速推理

### 模板匹配跟踪 (TemplateTrack)

- 基于模板匹配的跟踪算法
- 适用于特定场景的快速跟踪
- 低计算资源消耗

## 📡 网络通信

### ZMQ消息总线

- 支持多种消息模式（REQ/REP, PUB/SUB）
- 异步消息处理
- 跨进程通信

### UDP视频流

- 实时视频流传输
- H.264编码
- 可配置的网络参数

## 🔍 调试和日志

### 日志文件

- 系统日志: `logs/system.log`
- 模式特定日志: `logs/{mode}.log`
- 错误日志: `logs/error.log`

### 调试模式

```bash
# 启用详细调试
python entry.py --mode vis_fix --log-level DEBUG

# 查看实时日志
tail -f logs/vis_fix.log
```

## 🚨 故障排除

### 常见问题

1. **相机设备无法访问**

   ```bash
   # 检查设备权限
   ls -la /dev/video*
   sudo chmod 666 /dev/video*
   ```
2. **GStreamer管道错误**

   ```bash
   # 检查GStreamer插件
   gst-inspect-1.0 v4l2src
   gst-inspect-1.0 videoconvert
   ```
3. **网络连接问题**

   ```bash
   # 检查端口占用
   netstat -tulpn | grep :5599
   ```
4. **配置文件错误**

   ```bash
   # 验证TOML配置
   python -c "import toml; toml.load('configs/vis_fix/cam_config.toml')"
   ```
