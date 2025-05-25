import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams

# 支持汉字
rcParams['font.sans-serif'] = ['SimHei']


def multi_otsu_threshold(image, num_thresholds=1):
    """
    使用多个Otsu方法计算最佳阈值

    参数:
    image: 输入的单通道灰度图像
    num_thresholds: 需要计算的阈值数量

    返回:
    thresholds: 最佳阈值列表
    binary_image: 多阈值二值化后的图像
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

    max_variance = 0
    thresholds = [0] * num_thresholds

    # 生成所有可能的阈值组合
    def generate_threshold_combinations(prev_thresholds, start):
        if len(prev_thresholds) == num_thresholds:
            yield prev_thresholds
        else:
            for t in range(start, 256):
                new_thresholds = prev_thresholds + [t]
                yield from generate_threshold_combinations(new_thresholds, t + 1)

    # 遍历所有可能的阈值组合
    for threshold_combination in generate_threshold_combinations([], 1):
        w = []
        mu = []
        total_w = 0
        total_mu = 0

        # 计算每个区域的概率和均值
        for i in range(num_thresholds + 1):
            if i == 0:
                w.append(cdf[threshold_combination[i]])
            elif i < num_thresholds:
                w.append(cdf[threshold_combination[i]] - cdf[threshold_combination[i - 1]])
            else:
                w.append(1 - cdf[threshold_combination[-1]])

            if w[-1] > 0:
                if i == 0:
                    mu.append(np.sum(np.arange(threshold_combination[i] + 1) * hist_norm[:threshold_combination[i] + 1]) / w[-1])
                elif i < num_thresholds:
                    mu.append(np.sum(np.arange(threshold_combination[i - 1] + 1, threshold_combination[i] + 1) * hist_norm[threshold_combination[i - 1] + 1:threshold_combination[i] + 1]) / w[-1])
                else:
                    mu.append(np.sum(np.arange(threshold_combination[-1] + 1, 256) * hist_norm[threshold_combination[-1] + 1:]) / w[-1])
            else:
                mu.append(0)

            total_w += w[-1] * mu[-1]
            total_mu += w[-1] * mu[-1] ** 2

        # 计算类间方差
        variance = total_mu - total_w ** 2

        if variance > max_variance:
            max_variance = variance
            thresholds = threshold_combination

    # 根据多阈值进行多值化
    binary_image = np.zeros_like(gray)
    for i in range(num_thresholds + 1):
        if i == 0:
            binary_image[gray <= thresholds[i]] = 0
        elif i < num_thresholds:
            binary_image[(gray > thresholds[i - 1]) & (gray <= thresholds[i])] = 64 * i
        else:
            binary_image[gray > thresholds[-1]] = 255

    return thresholds, binary_image, hist


def main():
    # 图像路径
    image_path = '../resources/assets/exp7/701.png'
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print("无法加载图像，请检查路径!")
        exit(1)

    # 转换为RGB以便matplotlib显示
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 高斯滤波
    kernel_size = (5, 5)  # 高斯核大小
    sigma = 5.0  # 标准差
    blurred_image = cv2.GaussianBlur(image, kernel_size, sigma)
    blurred_image_rgb = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB)

    # 应用多Otsu阈值分割 - 原图
    thresholds_original, binary_image_original, hist_original = multi_otsu_threshold(image)

    # 应用多Otsu阈值分割 - 滤波后
    thresholds_blurred, binary_image_blurred, hist_blurred = multi_otsu_threshold(blurred_image)

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

    # 原始图像的多Otsu分割
    plt.subplot(233)
    plt.title(f'原图多Otsu分割 (阈值={thresholds_original})')
    plt.imshow(binary_image_original, cmap='gray')
    plt.axis('off')

    # 滤波后图像的多Otsu分割
    plt.subplot(236)
    plt.title(f'滤波后多Otsu分割 (阈值={thresholds_blurred})')
    plt.imshow(binary_image_blurred, cmap='gray')
    plt.axis('off')

    # 原始图像直方图
    plt.subplot(232)
    plt.title(f'原始图像直方图')
    plt.plot(hist_original, color='black')
    for t in thresholds_original:
        plt.axvline(x=t, color='r', linestyle='--')
    plt.xlabel('像素值')
    plt.ylabel('像素数量')

    # 滤波后图像直方图
    plt.subplot(235)
    plt.title(f'滤波后图像直方图')
    plt.plot(hist_blurred, color='black')
    for t in thresholds_blurred:
        plt.axvline(x=t, color='g', linestyle='--')
    plt.xlabel('像素值')
    plt.ylabel('像素数量')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
