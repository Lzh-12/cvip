import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 支持汉字
rcParams['font.sans-serif'] = ['SimHei']


def laplacian_filter(shape):
    h, w = shape
    u = np.fft.fftfreq(h).reshape(-1, 1)  # 列向量 (h, 1)
    v = np.fft.fftfreq(w).reshape(1, -1)  # 行向量 (1, w)

    # 频域拉普拉斯核：-4π²(u² + v²)
    laplacian = -4 * (np.pi ** 2) * (u ** 2 + v ** 2)
    return laplacian


def show_freq(freq_data):
    dft_shift = np.fft.fftshift(freq_data)
    # 步骤3：计算幅度谱
    magnitude = np.abs(dft_shift)
    # 对数变换增强显示效果（避免过大的数值）
    magnitude = np.log(magnitude + 1)
    # 归一化到 0 到 1
    magnitude = normalize(magnitude)
    return np.uint8(255.0 * magnitude)


def normalize(x, min_v=0, max_v=1.0):
    # 计算数组内的最小值和最大值
    min_val = np.min(x)
    v = (x - min_val) / (np.max(x) - min_val + 1e-8)
    return v * (max_v - min_v) + min_v


def show_image(image):
    image = normalize(image)
    return np.uint8(255.0 * image)


def main():
    # 图像路径
    image_path = '../resources/assets/exp4/407.png'
    # 读取图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image is None:
        print(f"无法读取图像: {image_path}")
        exit(1)

    freq_data = np.fft.fft2(image, axes=(0, 1))
    mask = laplacian_filter(image.shape[0:2])

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 4, 1)
    plt.imshow(show_image(image), cmap='gray')
    plt.title('原始图像')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(show_freq(freq_data), cmap='gray')
    plt.title('傅里叶变换频谱')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(show_freq(mask[:, :, np.newaxis]), cmap='gray')
    plt.title('拉普拉斯滤波核')
    plt.axis('off')

    # 图像锐化
    c = -1
    sharpened_image = freq_data + c * (freq_data * mask[:, :, np.newaxis])
    sharpened_image = np.real(np.fft.ifft2(sharpened_image, axes=(0, 1)))
    plt.subplot(1, 4, 4)
    sharpened_image = np.clip(sharpened_image, 0.0, 255.0)
    sharpened_image = np.uint8(sharpened_image)
    plt.imshow(sharpened_image, cmap='gray')
    plt.title('锐化图像')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()
