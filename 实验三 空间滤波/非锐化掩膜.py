import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 支持汉字显示
rcParams['font.sans-serif'] = ['SimHei']


def show_sharpening_process(image_path, kernel_size=(5, 5), sigma=1.0, amount=1.0):
    """
    显示图像锐化的完整过程：原始图像 -> 平滑图像 -> 高频分量 -> 锐化结果
    :param image_path: 图像路径
    :param kernel_size: 高斯核大小
    :param sigma: 高斯核标准差
    :param amount: 锐化强度
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return

    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 高斯平滑
    smoothed = cv2.GaussianBlur(gray, kernel_size, sigma)

    # 计算高频分量（原始 - 平滑）
    high_freq = np.float32(gray) - np.float32(smoothed)

    # 归一化高频分量以便显示
    high_freq_normalized = cv2.normalize(high_freq, None, 0, 255,
                                         cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 基于高频分量重建锐化图像
    # sharpened = original + amount * high_freq
    # 等价于: sharpened = (1+amount)*original - amount*smoothed
    sharpened = np.float32(gray) + amount * high_freq

    # 限制像素值范围并转换回uint8
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    # 创建显示窗口
    plt.figure(figsize=(20, 8))

    # 显示原始图像
    plt.subplot(141)
    plt.title("原始图像")
    plt.imshow(gray, cmap='gray')
    plt.axis('off')

    # 显示平滑图像
    plt.subplot(142)
    plt.title(f"高斯滤波图像 (σ={sigma})")
    plt.imshow(smoothed, cmap='gray')
    plt.axis('off')

    # 显示高频分量
    plt.subplot(143)
    plt.title("高频分量")
    plt.imshow(high_freq_normalized, cmap='gray')
    plt.axis('off')

    # 显示锐化结果
    plt.subplot(144)
    plt.title(f"锐化图像 (强度={amount})")
    plt.imshow(sharpened, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def main():
    # 替换为你的图像路径
    image_path = '../resources/assets/exp3/317.jpg'

    # 测试不同的锐化参数
    show_sharpening_process(image_path, sigma=1.0, amount=1.0)


if __name__ == '__main__':
    main()
