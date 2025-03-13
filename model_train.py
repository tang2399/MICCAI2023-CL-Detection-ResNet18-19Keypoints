import time
import cv2
import os
import torch
import numpy as np
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
                    except ValueError as e:
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

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 数据增强与归一化
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((256, 256)),
        T.ToTensor()
    ])

    # 超参数
    epochs, lr = 200, 5e-3

    model = ResNet18(num_keypoints=num_keypoints, image_width=1935, image_height=2400)
    model = model.to(device)
    # 数据集加载
    images, annotations = load_data(image_dir, label_dir)
    dataset = KeypointDataset(images, annotations, transform)
    batch_size = int(len(images) / 10)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # 损失函数与学习器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    total_loss, total_acc = [], []
    x = []
    start_time = time.time()

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
            point = []
            out = outputs.cpu().data.numpy()
            real = keypoints.cpu().data.numpy()
            for i in range(len(out)):
                for j in range(len(out[1])):
                    if len(point) == 2:
                        num_item += 1
                        distance = (point[0] ** 2 + point[1] ** 2) ** 0.5
                        if distance < 20:
                            running_acc += 1
                        point = []
                    else:
                        point.append(out[i][j] - real[i][j])

            running_loss += loss.item()

        using_time = time.time() - start_time
        using_s = int(using_time % 60)
        using_min = int(using_time // 60)

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(dataloader):.4f}"
              f" Acc: {running_acc / num_item:.4f},"
              f" UsingTime: {using_min}m{using_s}s")

        x.append(epoch + 1)
        total_loss.append(running_loss / len(dataloader))
        total_acc.append(running_acc / num_item)

    # 保存模型
    torch.save(model, "model.pth")
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
    plt.show()
