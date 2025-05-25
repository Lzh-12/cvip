import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import data
from matplotlib import rcParams


# 支持汉字
rcParams['font.sans-serif'] = ['SimHei']


def otsu_threshold(image):
    """
    使用Otsu方法计算最佳阈值

    参数:
    image: 输入的单通道灰度图像

    返回:
    threshold: 最佳阈值
    binary_image: 二值化后的图像
    hist: 图像直方图
    """
    # 确保图像是单通道灰度图
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 计算直方图
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_norm = hist.ravel() / hist.sum()

    # 计算累积分布函数和均值
    cdf = hist_norm.cumsum()
    mean = np.sum(np.arange(256) * hist_norm)

    # 初始化最大类间方差和最佳阈值
    max_variance = 0
    threshold = 0

    # 遍历所有可能的阈值(0-255)
    for t in range(256):
        # 计算前景和背景的概率
        w0 = cdf[t]
        w1 = 1 - w0

        if w0 == 0 or w1 == 0:
            continue

        # 计算前景和背景的均值
        mean0 = np.sum(np.arange(t + 1) * hist_norm[:t + 1]) / w0
        mean1 = np.sum(np.arange(t + 1, 256) * hist_norm[t + 1:]) / w1

        # 计算类间方差
        variance = w0 * w1 * (mean0 - mean1) ** 2

        # 更新最大方差和阈值
        if variance > max_variance:
            max_variance = variance
            threshold = t

    # 应用阈值进行二值化
    _, binary_image = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    return threshold, binary_image, hist


def main():
    # 图像路径
    image_path = '../resources/assets/exp7/703.png'
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print("无法加载图像，请检查路径!")
        exit(1)

    # 转换为RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 高斯滤波
    kernel_size = (5, 5)  # 高斯核大小
    sigma = 5.0  # 标准差
    blurred_image = cv2.GaussianBlur(image, kernel_size, sigma)
    blurred_image_rgb = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB)

    # 应用Otsu阈值分割 - 原图
    threshold_original, binary_image_original, hist_original = otsu_threshold(image)

    # 应用Otsu阈值分割 - 滤波后
    threshold_blurred, binary_image_blurred, hist_blurred = otsu_threshold(blurred_image)

    # 显示结果
    plt.figure(figsize=(18, 10))

    # 原始图像
    plt.subplot(231)
    plt.title('原始图像')
    plt.imshow(image_rgb)
    plt.axis('off')

    # 高斯滤波后图像
    plt.subplot(234)
    plt.title(f'高斯滤波后 (核大小={kernel_size}, σ={sigma})')
    plt.imshow(blurred_image_rgb)
    plt.axis('off')

    # 原始图像的Otsu分割
    plt.subplot(233)
    plt.title(f'原图Otsu分割 (阈值={threshold_original})')
    plt.imshow(binary_image_original, cmap='gray')
    plt.axis('off')

    # 滤波后图像的Otsu分割
    plt.subplot(236)
    plt.title(f'滤波后Otsu分割 (阈值={threshold_blurred})')
    plt.imshow(binary_image_blurred, cmap='gray')
    plt.axis('off')

    # 原始图像直方图
    plt.subplot(232)
    plt.title(f'原始图像直方图')
    plt.plot(hist_original, color='black')
    plt.axvline(x=threshold_original, color='r', linestyle='--', label=f'阈值={threshold_original}')
    plt.xlabel('像素值')
    plt.ylabel('像素数量')
    plt.legend()
    # plt.grid(True)

    # 滤波后图像直方图
    plt.subplot(235)
    plt.title(f'滤波后图像直方图')
    plt.plot(hist_blurred, color='black')
    plt.axvline(x=threshold_blurred, color='g', linestyle='--', label=f'阈值={threshold_blurred}')
    plt.xlabel('像素值')
    plt.ylabel('像素数量')
    plt.legend()
    # plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
