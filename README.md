# MICCAI 2023 CL Detection Keypoint Prediction with ResNet18



 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)



复现MICCAI 2023 CL-Detection比赛的简化实现方案，基于ResNet18的19个医学关键点检测模型，验证集MSE小于160。

## 核心特性

- 🚀 **轻量复现**：聚焦19关键点预测核心任务，简化比赛原始流程
- ⚡ **高效初始化**：`CoordinateInitialization.py` 实现均值初始化策略
- 📦 **模块化设计**：训练/推理/模型定义分离，便于二次开发
- 🏆 **达标性能**：在简化任务上稳定实现MSE <160像素
- 🖥️ **图形界面**：提供直观的GUI界面，方便图像选择和关键点预测
- 🛡️ **资源管理**：自动释放GPU资源，确保程序正常退出时清理内存
- 💾 **最佳模型保存**：训练过程中保存验证指标最佳的模型，而非最后一个模型
- <img src="https://raw.githubusercontent.com/tang2399/MICCAI2023-CL-Detection-ResNet18-19Keypoints/master/images/Train.png"       alt="Loss and ACC"       style="float: left; margin-right: 20px;"       width="500"/>

## 快速开始
### 环境安装
```bash
git clone https://github.com/tang2399/MICCAI2023-CL-Detection-ResNet18-19Keypoints.git
cd MICCAI2023-CL-Detection-ResNet18-19Keypoints
pip install -r requirements.txt
```

### 数据准备

按以下结构存放数据：

```
master/
├── dataset/
│   ├── train/
│   │   ├── images/      # 训练图像 (*.bmp)
│   │   └── labels/ 	 # txt标注文件
│   └── apply/
│       └── images/      # 测试图像
├── CoordinateInitialization.py
├── model.py
├── model_apply.py
├── model_train.py
├── gui_model_apply.py   # 图形界面应用
```

## 性能验证

| 指标    | 数值  | 条件                                          |
| ------- | ----- | --------------------------------------------- |
| **MSE** | 158.2 | 1935x2400分辨率                               |
| **ACC** | 65%   | 1935x2400分辨率时，预测点在目标点20像素范围内 |

## 使用方法

### 命令行应用

使用命令行方式运行预测：

```bash
# 修改model_apply.py中的图像路径后运行
python model_apply.py
```

### 图形界面应用

使用GUI界面进行预测更加直观：

```bash
# 启动图形界面应用
python gui_model_apply.py
```

GUI应用提供以下功能：
- 图像浏览和选择
- 一键运行关键点预测
- 可视化显示预测结果
- 保存预测结果为图像和CSV文件
- 自动资源管理，确保程序退出时释放GPU资源

### 训练模型

训练自己的模型：

```bash
# 开始训练，会自动保存最佳模型为best_model.pth
python model_train.py
```

## 资源管理

本项目实现了完善的资源管理机制：
- 自动释放GPU内存缓存
- 正常和异常退出时都能清理资源
- 支持信号处理（如Ctrl+C）时的资源释放
- 关闭所有matplotlib图形

## 常见问题

❓ **标注文件格式要求？** txt文件需包含 `x1,y1,...,x19,y19` 格式的绝对坐标，每个坐标点占一行

❓ **如何使用GUI应用？** 运行`gui_model_apply.py`，点击"Browse"选择图像，然后点击"Run Prediction"进行预测

❓ **为什么选择保存最佳模型？** 训练过程中会根据验证损失自动保存性能最佳的模型，避免过拟合问题