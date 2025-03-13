# MICCAI 2023 CL Detection Keypoint Prediction with ResNet18



 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/) 



复现MICCAI 2023 CL-Detection比赛的简化实现方案，基于ResNet18的19个医学关键点检测模型，验证集MSE **<160像素 **

## 核心特性 

- 🚀 **轻量复现**：聚焦19关键点预测核心任务，简化比赛原始流程 
- ⚡ **高效初始化**：`CoordinateInitialization.py` 实现均值初始化策略 
- 📦 **模块化设计**：训练/推理/模型定义分离，便于二次开发 
- 🏆 **达标性能**：在简化任务上稳定实现MSE <160像素 
- <img src="https://raw.githubusercontent.com/tang2399/MICCAI2023-CL-Detection-ResNet18-19Keypoints/master/images/Train.png"       alt="Loss and ACC"       style="float: left; margin-right: 20px;"       width="500"/>

## 快速开始
### 环境安装 
```bash
git clone https://github.com/tang2399/MICCAI2023-CL-Detection-ResNet18-Keypoints.git
cd MICCAI2023-CL-Detection-ResNet18-Keypoints pip install -r requirements.txt
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
```

## 性能验证

| 指标    | 数值  | 条件             |
| ------- | ----- | ---------------- |
| **MSE** | 158.2 | 1935x2400分辨率  |
| **ACC** | 65%   | 目标点20像素点内 |

## 常见问题

❓ **标注文件格式要求？** txt文件需包含 `x1,y1,...,x19,y19` 格式的绝对坐标，每个坐标点占一行