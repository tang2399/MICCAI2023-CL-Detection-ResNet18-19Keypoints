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
        initialX = [810.4222222222222, 1396.5055555555555, 1297.9833333333333, 597.8083333333333,
                    1387.3944444444444, 1348.9555555555555, 1344.8333333333333, 1280.6888888888889,
                    1324.5222222222221, 720.9666666666667, 1423.9305555555557, 1442.2416666666666,
                    1552.7194444444444, 1521.5388888888888, 1500.9805555555556, 1457.9277777777777,
                    984.45, 1409.6138888888888, 674.3777777777777]
        initialY = [1037.7972222222222, 966.6833333333333, 1213.1916666666666, 1203.9472222222223,
                    1507.9138888888888, 1869.638888888889, 1994.525, 2047.1666666666667, 
                    2032.263888888889, 1730.7944444444445, 1695.9277777777777, 1709.5083333333334, 
                    1595.4444444444443, 1829.85, 1489.7083333333333, 1984.111111111111, 1433.3916666666667, 
                    1449.0694444444443, 1331.4916666666666]
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
