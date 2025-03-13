from model_train import label_dir, num_keypoints
import os
import numpy as np

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
                    x, y = map(lambda v: float(v.replace(',', '.')), line.strip().split(','))  # 按逗号分割
                    keypoints.append([x, y])
                except ValueError as e:
                    print(f"Error processing line in {txt_path}: {line.strip()}")
                    continue

        # 将关键点转换为 NumPy 数组
        annotations.append(np.array(keypoints))

x, y = [0] * num_keypoints, [0] * num_keypoints
for i in range(len(annotations)):  # 遍历所有图片
    for j in range(num_keypoints):  # 遍历所有关键点
        for k in range(2):  # 遍历x，y
            if k == 0:
                x[j] += annotations[i][j][k]
            else:
                y[j] += annotations[i][j][k]

for i in range(num_keypoints):
    x[i] /= len(annotations)
    y[i] /= len(annotations)
print("数据集的关键点的X值平均值为：", x)
print("数据集的关键点的Y值平均值为：", y)

