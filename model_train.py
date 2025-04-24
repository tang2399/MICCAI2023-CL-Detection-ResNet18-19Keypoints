import time
import cv2
import os
import torch
import numpy as np
import gc
import signal
import sys
import atexit
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from model import ResNet18, num_keypoints
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt


# 读取所有标注文件
def load_data(image_dir, label_dir):
    images = []
    annotations = []

    for file_name in os.listdir(label_dir):
        if file_name.endswith(".txt"):  # 只处理 .txt 文件
            txt_path = os.path.join(label_dir, file_name)

            # 读取 .txt 文件前 19 行
            keypoints = []
            with open(txt_path, "r") as f:
                lines = f.readlines()
                for line in lines[:19]:  # 只取前 19 行
                    if not line.strip():  # 跳过空行
                        continue
                    try:
                        x, y = map(lambda v: float(v.replace(',', '.')),
                                   line.strip().split(','))  # 按逗号分割
                        keypoints.append([x, y])
                    except ValueError:
                        print(f"Error processing line in {txt_path}: {line.strip()}")
                        continue

            # 将关键点转换为 NumPy 数组
            annotations.append(np.array(keypoints))

            # 对应的图像路径
            image_name = os.path.splitext(file_name)[0] + ".bmp"
            image_path = os.path.join(image_dir, image_name)
            images.append(image_path)

    return images, annotations


# 数据预处理
class KeypointDataset(Dataset):
    def __init__(self, images, annotations, transform=None, output_size=(256, 256)):
        self.images = images
        self.annotations = annotations
        self.transform = transform
        self.output_size = output_size  # 目标输出尺寸

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 加载图像
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape  # 原始图像尺寸

        # 获取关键点并归一化
        keypoints = self.annotations[idx].copy()  # 确保不修改原始数据
        # 计算缩放比例
        new_h, new_w = self.output_size
        # 缩放图像
        image = cv2.resize(image, (new_w, new_h))
        # 应用图像变换
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(keypoints.flatten(), dtype=torch.float32)


image_dir = "dataset/train/images"
label_dir = "dataset/train/labels"

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

    print("Cleanup completed")

# 处理信号的函数
def signal_handler(sig, frame):
    print(f"\nReceived signal {sig}, performing cleanup before exit...")
    cleanup_resources()
    sys.exit(0)

if __name__ == '__main__':
    print("Starting training script...")

    # 注册信号处理器，用于处理Ctrl+C等中断信号
    signal.signal(signal.SIGINT, signal_handler)   # 处理Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # 处理终止信号

    # 注册正常退出时的清理函数
    atexit.register(cleanup_resources)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 数据增强与归一化
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((256, 256)),
        T.ToTensor()
    ])

    # 超参数
    epochs, lr = 200, 5e-3
    print(f"Training for {epochs} epochs with learning rate {lr}")

    model = ResNet18(num_keypoints=num_keypoints, image_width=1935, image_height=2400)
    model = model.to(device)
    print("Model initialized")

    # 数据集加载
    print(f"Loading data from {image_dir} and {label_dir}")
    images, annotations = load_data(image_dir, label_dir)
    print(f"Loaded {len(images)} images and {len(annotations)} annotation sets")

    dataset = KeypointDataset(images, annotations, transform)
    batch_size = max(1, int(len(images) / 10))
    print(f"Batch size: {batch_size}")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # 损失函数与学习器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    total_loss, total_acc = [], []
    x = []
    start_time = time.time()

    # 初始化最佳模型跟踪变量
    best_loss = float('inf')
    best_model_state = None

    try:
        # 训练
        for epoch in range(epochs):
            model.train()

            running_loss = 0.0
            running_acc = 0.0
            num_item = 0

            for batch_idx, (images, keypoints) in enumerate(dataloader):

                images = images.to(device).float()
                keypoints = keypoints.to(device).float()

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, keypoints)
                loss.backward()
                optimizer.step()

                # 计算准确度
                out = outputs.cpu().data.numpy()
                real = keypoints.cpu().data.numpy()

                # 打印输出和真实值的形状，以便调试
                if batch_idx == 0 and epoch == 0:
                    print(f"Output shape: {out.shape}, Real shape: {real.shape}")

                # 重塑数组以便于处理关键点对
                batch_size = out.shape[0]
                try:
                    out_reshaped = out.reshape(batch_size, num_keypoints, 2)
                    real_reshaped = real.reshape(batch_size, num_keypoints, 2)

                    # 计算每个关键点的欧氏距离
                    for i in range(batch_size):
                        for j in range(num_keypoints):
                            num_item += 1
                            # 计算欧氏距离
                            pred_point = out_reshaped[i, j]
                            gt_point = real_reshaped[i, j]
                            distance = np.sqrt(np.sum((pred_point - gt_point) ** 2))

                            # 如果距离小于阈值，则认为预测正确
                            if distance < 20:  # 可以根据需要调整阈值
                                running_acc += 1

                    # 打印第一个批次的一些距离信息，以便调试
                    if batch_idx == 0 and epoch == 0:
                        for j in range(min(5, num_keypoints)):  # 只打印前5个关键点
                            pred_point = out_reshaped[0, j]
                            gt_point = real_reshaped[0, j]
                            distance = np.sqrt(np.sum((pred_point - gt_point) ** 2))
                            print(f"Keypoint {j}: Pred {pred_point}, GT {gt_point}, Distance {distance:.2f}")
                except Exception as e:
                    print(f"Error in reshaping or distance calculation: {e}")
                    print(f"Output shape: {out.shape}, Real shape: {real.shape}")
                    # 如果重塑失败，使用原始方法计算准确度
                    running_acc = 0  # 重置准确度计数
                    num_item = 1  # 避免除零错误

                running_loss += loss.item()

            # 计算当前epoch的平均损失
            epoch_loss = running_loss / len(dataloader)
            epoch_acc = running_acc / num_item

            using_time = time.time() - start_time
            using_s = int(using_time % 60)
            using_min = int(using_time // 60)

            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}"
                  f" Acc: {epoch_acc:.4f},"
                  f" UsingTime: {using_min}m{using_s}s")

            # 检查是否是最佳模型
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_state = model.state_dict().copy()
                print(f"New best model saved with loss: {best_loss:.4f}")
                # 保存当前最佳模型
                torch.save(model, "best_model.pth")

            x.append(epoch + 1)
            total_loss.append(epoch_loss)
            total_acc.append(epoch_acc)

        # 如果需要，恢复最佳模型状态
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"Restored best model with loss: {best_loss:.4f}")

        # 保存最终模型（这是最佳模型）
        torch.save(model, "model.pth")
        print("Best model saved as 'model.pth'")

    except Exception as e:
        print(f"Training interrupted by exception: {e}")
        # 如果有最佳模型，尝试保存它
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            torch.save(model, "emergency_best_model.pth")
            print("Emergency save of best model completed")
        raise  # 重新抛出异常以便查看完整的错误信息

    finally:
        # 即使发生异常，也会执行清理操作
        # 注意：这里不需要显式调用cleanup_resources()，因为它已经通过atexit注册
        # 但如果需要在这里执行额外的清理操作，可以添加
        pass

    # 绘制训练结果图表
    try:
        photo_save_path = "outputs/training_results.png"
        plt.rcParams['font.sans-serif'] = ['SimHei']
        # 创建图表和第一个纵坐标轴
        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color=color)
        ax1.plot(x, total_loss, color=color, marker='o', label='Loss', markersize=2)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend(loc='upper left')
        # 添加第二个纵坐标轴
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Acc', color=color)
        ax2.plot(x, total_acc, color=color, marker='o', label='Acc', markersize=2)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc='upper right')
        # 显示图表
        plt.title('Loss And Acc')
        plt.savefig(photo_save_path)  # 保存图表到文件
        print("Training results plot saved to '%s'" % photo_save_path)
        plt.show()
    except Exception as e:
        print(f"Error creating or displaying plot: {e}")
        # 继续尝试保存图表
        try:
            plt.savefig(photo_save_path)
            print("Training results plot saved to '%s' despite display error" % photo_save_path)
        except:
            print("Failed to save training results plot")
