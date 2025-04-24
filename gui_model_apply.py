import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import torchvision.transforms as T
from PIL import Image
from model import classes

class KeypointDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Keypoint Detection Application")
        self.root.geometry("900x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # 设置设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 数据转换
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 256)),
            T.ToTensor()
        ])

        # 模型加载
        self.model_path = "model.pth"

        try:
            # 加载模型
            self.model = torch.load(self.model_path, map_location=self.device)

            # 将模型移至设备并设置为评估模式
            self.model.to(self.device)
            self.model.eval()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}\nPlease check if the model file exists and is valid.")
            self.root.destroy()
            return

        # 变量
        self.current_image_path = None
        self.current_image = None
        self.current_results = None

        # 创建用户界面
        self.create_ui()

    def create_ui(self):
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 左侧面板 - 控制区
        left_panel = ttk.Frame(main_frame, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # 文件选择
        ttk.Label(left_panel, text="Image Selection", font=("Arial", 12, "bold")).pack(pady=(0, 10), anchor=tk.W)

        select_frame = ttk.Frame(left_panel)
        select_frame.pack(fill=tk.X, pady=(0, 20))

        self.path_var = tk.StringVar()
        ttk.Entry(select_frame, textvariable=self.path_var, state="readonly").pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(select_frame, text="Browse", command=self.browse_file).pack(side=tk.RIGHT, padx=(5, 0))

        # 预测按钮
        ttk.Button(left_panel, text="Run Prediction", command=self.run_prediction).pack(fill=tk.X, pady=(0, 20))

        # 保存结果按钮
        ttk.Button(left_panel, text="Save Results", command=self.save_results).pack(fill=tk.X)

        # 进度条
        self.progress_var = tk.DoubleVar()
        ttk.Label(left_panel, text="Progress:").pack(anchor=tk.W, pady=(20, 5))
        self.progress_bar = ttk.Progressbar(left_panel, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X)

        # 状态
        ttk.Label(left_panel, text="Status:").pack(anchor=tk.W, pady=(20, 5))
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(left_panel, textvariable=self.status_var).pack(anchor=tk.W)

        # 右侧面板 - 显示区
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 图像预览
        ttk.Label(right_panel, text="Image Preview", font=("Arial", 12, "bold")).pack(pady=(0, 10), anchor=tk.W)

        self.preview_frame = ttk.Frame(right_panel, relief=tk.SUNKEN, borderwidth=1)
        self.preview_frame.pack(fill=tk.BOTH, expand=True)

        # 创建matplotlib图形
        self.fig = plt.Figure(figsize=(6, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.preview_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 初始显示
        self.ax.set_title("No image selected")
        self.fig.tight_layout()
        self.canvas.draw()

    def browse_file(self):
        filetypes = [
            ("Image files", "*.bmp;*.jpg;*.jpeg;*.png"),
            ("All files", "*.*")
        ]

        # 确定初始目录
        initial_dirs = [
            "dataset/apply/images",
            "dataset/apply",
            "dataset/train/images",
            "dataset/train",
            "dataset",
            "."
        ]

        initial_dir = next((d for d in initial_dirs if os.path.exists(d)), ".")

        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=filetypes,
            initialdir=initial_dir
        )

        if file_path:
            # 检查文件是否存在且可读
            if not os.path.exists(file_path):
                messagebox.showerror("Error", f"File does not exist: {file_path}")
                return

            if not os.access(file_path, os.R_OK):
                messagebox.showerror("Error", f"File is not readable: {file_path}")
                return

            # 检查文件大小
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB

            if file_size > 100:  # If file is larger than 100MB
                if not messagebox.askyesno("Warning", f"Selected file is very large ({file_size:.2f} MB). This may cause performance issues. Continue?"):
                    return

            self.current_image_path = file_path
            self.path_var.set(file_path)
            self.status_var.set(f"Image selected: {os.path.basename(file_path)}")
            self.display_image(file_path)

    def display_image(self, image_path):
        try:
            # 清除之前的图
            self.ax.clear()

            # 加载图像
            try:
                # 尝试使用OpenCV加载
                image = cv2.imread(image_path)
                if image is None:
                    # 如果OpenCV失败，尝试使用PIL
                    pil_image = Image.open(image_path)
                    image = np.array(pil_image)
                    # 如果图像是RGB格式，转换为BGR以与OpenCV保持一致
                    if len(image.shape) == 3 and image.shape[2] == 3:
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            except Exception:
                raise ValueError(f"Unable to load image: {image_path}")

            if image is None or image.size == 0:
                raise ValueError(f"Loaded image is empty: {image_path}")

            # 将BGR转换为RGB以便显示
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.current_image = image

            # 显示图像
            self.ax.imshow(image_rgb)
            self.ax.set_title(f"Image: {os.path.basename(image_path)}")
            self.ax.axis('off')

            # 更新画布
            self.fig.tight_layout()
            self.canvas.draw()

            # 更新状态
            self.status_var.set(f"Image loaded: {os.path.basename(image_path)}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to display image: {str(e)}")
            self.status_var.set("Error displaying image")

    def run_prediction(self):
        if self.current_image_path is None or self.current_image is None:
            messagebox.showwarning("Warning", "Please select an image first")
            return

        self.status_var.set("Running prediction...")
        self.progress_var.set(10)
        self.root.update_idletasks()

        try:
            # 准备图像
            self.progress_var.set(20)
            self.root.update_idletasks()

            # 转换图像
            input_image = self.transform(self.current_image).unsqueeze(0).to(self.device)

            self.progress_var.set(40)
            self.root.update_idletasks()

            # 运行预测
            self.progress_var.set(50)
            self.root.update_idletasks()

            with torch.no_grad():
                model_output = self.model(input_image)
                outputs = model_output.cpu().numpy().reshape(-1, 2)

            self.progress_var.set(70)
            self.root.update_idletasks()

            # 存储结果
            self.current_results = outputs

            # 显示结果
            self.display_results(self.current_image, outputs)

            self.progress_var.set(100)
            self.status_var.set("Prediction completed")

        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
            self.status_var.set("Prediction failed")
            self.progress_var.set(0)

    def display_results(self, image, keypoints):
        # 清除之前的图
        self.ax.clear()

        # 将BGR转换为RGB以便显示
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 显示图像
        self.ax.imshow(image_rgb)

        # 提取x和y坐标
        x, y = [], []
        for i in keypoints:
            x.append(i[0])
            y.append(i[1])

        # 绘制关键点
        self.ax.scatter(x, y, color='blue', s=10)

        # 添加标签
        for i in range(len(x)):
            self.ax.annotate(classes[i], (x[i], y[i]), textcoords="offset points",
                            xytext=(0, 5), ha='center', fontsize=8, color='red')

        self.ax.set_title(f"Prediction Results: {os.path.basename(self.current_image_path)}")
        self.ax.axis('off')

        # 更新画布
        self.fig.tight_layout()
        self.canvas.draw()

    def save_results(self):
        if self.current_results is None or self.current_image is None:
            messagebox.showwarning("Warning", "No prediction results to save")
            return

        # 请求保存路径
        save_path = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*")
            ],
            initialdir=os.path.dirname(self.current_image_path)
        )

        if not save_path:
            return

        try:
            # 保存图形
            self.fig.savefig(save_path, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Success", f"Results saved to {save_path}")
            self.status_var.set(f"Results saved to {os.path.basename(save_path)}")

            # 同时保存关键点数据为CSV
            csv_path = os.path.splitext(save_path)[0] + ".csv"
            with open(csv_path, 'w') as f:
                f.write("Keypoint,X,Y\n")
                for i, (x, y) in enumerate(self.current_results):
                    f.write(f"{classes[i]},{x},{y}\n")

            messagebox.showinfo("Success", f"Keypoint data saved to {csv_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")

    def on_closing(self):
        """Handle window closing event"""
        try:
            # 清理资源
            if hasattr(self, 'model') and self.model is not None:
                # 将模型移动到CPU以便删除以释放GPU内存
                if self.device == "cuda":
                    self.model.to("cpu")

                # 删除模型引用
                del self.model

                # 如果可用，清理CUDA缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            plt.close('all')  # 关闭所有matplotlib图形

        except Exception:
            pass
        finally:
            self.root.destroy()


if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = KeypointDetectionApp(root)
        root.mainloop()
    except Exception:
        pass
    finally:
        # 确保GPU资源被释放
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 关闭所有matplotlib图形
        plt.close('all')
