import cv2
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T
from model import classes


# 加载和预处理图像
def predict(model, image_path):
    model.eval()
    with torch.no_grad():
        image = cv2.imread(image_path)
        h, w, _ = image.shape
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

        plt.figure(figsize=(6, 8))
        plt.imshow(image)
        plt.scatter(x, y, color='blue', s=3)
        for i in range(len(x)):
            plt.annotate(classes[i], (x[i], y[i]), textcoords="offset points", xytext=(0, 2), ha='center',
                         fontsize=5, color='red')
        plt.show()


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 数据增强与归一化
    transform = T.Compose([T.ToPILImage(), T.Resize((256, 256)), T.ToTensor()])

    model = torch.load("model.pth")
    image = "dataset/apply/351.bmp"
    predict(model, image)
