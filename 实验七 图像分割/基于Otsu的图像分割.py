import cv2
import numpy as np
from matplotlib import pyplot as plt
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

    return threshold, binary_image


def main():
    """主函数，演示Otsu图像分割效果"""
    # 读取图像
    image = cv2.imread('../resources/assets/exp7/701.png')
    if image is None:
        print("无法加载图像，请检查路径!")
        return

    # 转换为RGB以便matplotlib显示
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 应用Otsu阈值分割
    threshold, binary_image = otsu_threshold(image)

    # 计算使用OpenCV内置Otsu方法的结果作对比
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image_cv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 显示结果
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.title('原始图像')
    plt.imshow(image_rgb)
    plt.axis('off')

    plt.subplot(132)
    plt.title(f'自定义Otsu分割 (阈值={threshold})')
    plt.imshow(binary_image, cmap='gray')
    plt.axis('off')

    plt.subplot(133)
    plt.title(f'OpenCV内置Otsu分割 (阈值={_})')
    plt.imshow(binary_image_cv, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()