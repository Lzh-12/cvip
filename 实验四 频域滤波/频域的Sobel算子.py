import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 支持汉字
rcParams['font.sans-serif'] = ['SimHei']


# Sobel算子滤波
def frequency_domain_sobel(img):
    # 转换图像到频域
    f = np.fft.fft2(img)  # 对图像进行二维傅里叶变换
    # 低频成分移到频谱的中心
    fshift = np.fft.fftshift(f)

    rows, cols = img.shape

    # 创建频域 Sobel 算子
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)  # 水平方向
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)  # 垂直方向

    # 对 Sobel 算子进行零填充扩展到与图像相同的尺寸再频域处理
    sobel_x_padded = np.zeros((rows, cols), dtype=np.float32)
    sobel_x_padded[:sobel_x.shape[0], :sobel_x.shape[1]] = sobel_x
    sobel_y_padded = np.zeros((rows, cols), dtype=np.float32)
    sobel_y_padded[:sobel_y.shape[0], :sobel_y.shape[1]] = sobel_y

    # 将 Sobel 算子转换到频域
    sobel_x_freq = np.fft.fft2(sobel_x_padded)
    sobel_x_freq = np.fft.fftshift(sobel_x_freq)  # 将算子的零频分量移到中心
    sobel_y_freq = np.fft.fft2(sobel_y_padded)
    sobel_y_freq = np.fft.fftshift(sobel_y_freq)

    # 应用频域 Sobel 算子
    gx = fshift * sobel_x_freq
    gy = fshift * sobel_y_freq

    # 转换回空间域
    gx_back = np.fft.ifft2(np.fft.ifftshift(gx))
    gy_back = np.fft.ifft2(np.fft.ifftshift(gy))
    gx_back = np.abs(gx_back)
    gy_back = np.abs(gy_back)

    # 计算梯度幅值
    return f, np.fft.fftshift(f), sobel_x_freq, sobel_y_freq, gx, gy, gx_back, gy_back, np.sqrt(gx_back ** 2 + gy_back ** 2)


# 主函数
def main():
    # 图像路径
    image_path = '../resources/assets/exp4/403.jpg'

    # 读取图像
    image = cv2.imread(image_path, 0)  # 以灰度模式读取图像
    if image is None:
        print(f"无法读取图像: {image_path}")
        exit(1)

    # 应用频域 Sobel 算子进行边缘提取
    f, fshift, sobel_x_freq, sobel_y_freq, gx, gy, gx_back, gy_back, edge_sobel = frequency_domain_sobel(image)
    # 计算幅度谱
    magnitude_spectrum = 20 * np.log(1 + np.abs(fshift))
    gx_magnitude = 20 * np.log(1 + np.abs(gx))
    gy_magnitude = 20 * np.log(1 + np.abs(gy))

    plt.figure(figsize=(20, 10))

    # 原始图像
    plt.subplot(2, 5, 1)
    plt.imshow(image, cmap='gray')
    plt.title('原始图像')
    plt.xticks([]), plt.yticks([])

    # 原始图像的傅里叶变换幅度谱
    plt.subplot(2, 5, 6)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('原始图像的傅里叶变换幅度谱')
    plt.xticks([]), plt.yticks([])

    # 水平方向Sobel算子频域图
    plt.subplot(2, 5, 2)
    plt.imshow(20 * np.log(1 + np.abs(sobel_x_freq)), cmap='gray')
    plt.title('水平方向Sobel算子频域图')
    plt.xticks([]), plt.yticks([])

    # 垂直方向Sobel算子频域图
    plt.subplot(2, 5, 7)
    plt.imshow(20 * np.log(1 + np.abs(sobel_y_freq)), cmap='gray')
    plt.title('垂直方向Sobel算子频域图')
    plt.xticks([]), plt.yticks([])

    # 水平方向滤波结果频域图
    plt.subplot(2, 5, 3)
    plt.imshow(gx_magnitude, cmap='gray')
    plt.title('水平方向滤波结果频域图')
    plt.xticks([]), plt.yticks([])

    # 垂直方向滤波结果频域图
    plt.subplot(2, 5, 8)
    plt.imshow(gy_magnitude, cmap='gray')
    plt.title('垂直方向滤波结果频域图')
    plt.xticks([]), plt.yticks([])

    # 水平方向滤波结果
    plt.subplot(2, 5, 4)
    plt.imshow(gx_back, cmap='gray')
    plt.title('水平方向滤波结果')
    plt.xticks([]), plt.yticks([])

    # 垂直方向滤波结果
    plt.subplot(2, 5, 9)
    plt.imshow(gy_back, cmap='gray')
    plt.title('垂直方向滤波结果')
    plt.xticks([]), plt.yticks([])

    # Sobel边缘提取后的图像
    plt.subplot(2, 5, 5)
    plt.imshow(edge_sobel, cmap='gray')
    plt.title('频域Sobel滤波图像')
    plt.xticks([]), plt.yticks([])

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
