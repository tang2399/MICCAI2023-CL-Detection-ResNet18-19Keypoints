import torch.nn as nn
import torchvision.models as models
import torch


class ResNet18(nn.Module):
    def __init__(self, num_keypoints, image_width, image_height):
        super(ResNet18, self).__init__()
        self.num_keypoints = num_keypoints
        self.image_width = image_width
        self.image_height = image_height

        # 使用预训练的 ResNet18 作为特征提取器
        self.backbone = models.resnet18(pretrained=True)

        # 替换最后一层为关键点预测层
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_keypoints * 2)

        # 初始化关键点的初始位置（如图像中心点）
        self._initialize_keypoints()

    def forward(self, x):
        return self.backbone(x)

    def _initialize_keypoints(self):
        # 计算取得的初始化关键点位置
        initialX = [806.5666666666667, 1390.3761904761905, 1291.9666666666667, 596.2619047619048, 1379.2095238095237,
                    1348.6571428571428, 1347.8095238095239, 1283.4333333333334, 1328.1857142857143, 718.847619047619,
                    1420.509523809524, 1434.952380952381, 1551.2904761904763, 1525.2095238095237, 1496.6190476190477,
                    1460.4476190476191, 979.7190476190476, 1401.952380952381, 672.1904761904761]
        initialY = [1038.942857142857, 967.9761904761905, 1213.5714285714287, 1205.452380952381, 1503.7190476190476,
                    1868.0428571428572, 1989.8, 2045.347619047619, 2028.642857142857, 1730.0190476190476,
                    1693.1619047619047, 1706.0428571428572, 1599.2190476190476, 1823.557142857143, 1488.1238095238095,
                    1978.9380952380952, 1430.952380952381, 1447.7047619047619, 1329.2428571428572]
        initial_positions = []
        for i in range(self.num_keypoints):
            initial_positions.extend([initialX[i], initialY[i]])  # 每个关键点的 (x, y)

        # 将初始位置写入最后一层的偏置
        with torch.no_grad():
            self.backbone.fc.bias.copy_(torch.tensor(initial_positions, dtype=torch.float32))


num_keypoints = 19
classes = ['S', 'N', 'Or', 'P', 'A', 'B', 'Pog', 'Me', 'Gn',
           'Go', 'L1', 'U1', 'Ls', 'Li', 'Sn', 'Pog1', 'PNS', 'ANS', 'Ar']

if __name__ == '__main__':
    print('Hi, I am ResNet18')
