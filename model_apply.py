import cv2
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T
import os
import gc
import signal
import sys
import atexit
from model import classes


# 定义清理函数，用于释放资源
def cleanup_resources():
    print("Cleaning up resources...")
    # 释放GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU memory cache cleared")

    # 强制垃圾回收
    gc.collect()
    print("Garbage collection performed")

    # 关闭所有matplotlib图形
    plt.close('all')
    print("Closed all matplotlib figures")

    print("Cleanup completed")


# 处理信号的函数
def signal_handler(sig, frame):
    print(f"\nReceived signal {sig}, performing cleanup before exit...")
    cleanup_resources()
    sys.exit(0)


# 加载和预处理图像
def predict(model, image_path, transform):
    model.eval()
    with torch.no_grad():
        # 检查图像路径是否存在
        if not os.path.exists(image_path):
            print(f"错误: 图像文件 '{image_path}' 不存在")
            return

        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                print(f"错误: 无法读取图像 '{image_path}'")
                return

            # 获取设备
            device = next(model.parameters()).device

            # 图像处理
            input_image = transform(image).unsqueeze(0).to(device)

            # 预测关键点
            outputs = model(input_image).cpu().numpy().reshape(-1, 2)
            # 格式转换
            outputs = outputs * [1, 1]
            print(outputs)

            # 获取关键点列表
            x, y = [], []
            for i in outputs:
                x.append(i[0])
                y.append(i[1])

            # 显示结果
            plt.figure(figsize=(6, 8))
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # OpenCV读取的是BGR格式，转换为RGB以正确显示
            plt.scatter(x, y, color='blue', s=3)
            for i in range(len(x)):
                plt.annotate(classes[i], (x[i], y[i]), textcoords="offset points", xytext=(0, 2), ha='center',
                             fontsize=5, color='red')
            plt.show()

        except Exception as e:
            print(f"预测过程中发生错误: {e}")
            raise


if __name__ == '__main__':
    # 注册信号处理器，用于处理Ctrl+C等中断信号
    signal.signal(signal.SIGINT, signal_handler)   # 处理Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # 处理终止信号

    # 注册正常退出时的清理函数
    atexit.register(cleanup_resources)

    print("Starting model application...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 数据增强与归一化
    transform = T.Compose([T.ToPILImage(), T.Resize((256, 256)), T.ToTensor()])

    model = None
    try:
        # 加载模型
        print("Loading model...")
        model = torch.load("model.pth")
        model = model.to(device)

        # 设置图像路径
        image_path = "dataset/apply/images/361.bmp"
        print(f"Processing image: {image_path}")

        # 运行预测
        predict(model, image_path, transform)

    except Exception as e:
        print(f"Error occurred: {e}")

    finally:
        # 清理模型资源
        if model is not None and device == "cuda":
            # 将模型移到CPU以便删除以释放GPU内存
            model.to("cpu")
            del model

        # 注意：不需要显式调用cleanup_resources()，因为它已经通过atexit注册
