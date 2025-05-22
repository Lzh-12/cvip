import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 支持汉字
rcParams['font.sans-serif'] = ['SimHei']


# 创建高斯低通滤波器
def create_filter(img, D0):
    rows, cols = img.shape
    mid_row, mid_col = rows // 2, cols // 2  # 中心坐标

    # 创建高斯低通滤波器
    H = np.zeros((rows, cols), np.float32)
    for i in range(rows):
        for j in range(cols):
            # 到中心点的距离
            D = np.sqrt((i - mid_row) ** 2 + (j - mid_col) ** 2)
            # 高斯低通滤波器
            H[i, j] = np.exp(-(D ** 2) / (2 * (D0 ** 2)))

    return H


# 高斯高通滤波
def gaussian_highpass_filter(img, D0):
    # 图像空间域->频域
    f = np.fft.fft2(img)

    # 创建高斯高通滤波器
    H = create_filter(img, D0)

    # 滤波器空间域->频域
    H = np.fft.fftshift(H)

    # 卷积
    f_shift_filtered = f * H

    # 图像频域->空间域
    img_back = np.fft.ifft2(f_shift_filtered)
    img_back = np.abs(img_back)

    return H, f_shift_filtered, img_back


# 主函数
def main():
    # 图像路径
    image_path = '../resources/assets/exp4/401.png'

    # 读取图像
    image = cv2.imread(image_path, 0)  # 以灰度模式读取图像
    if image is None:
        print(f"无法读取图像: {image_path}")
        exit(1)

    # 高斯高通滤波器的截止频率
    D0 = 160
    # 应用高斯高通滤波器进行边缘提取
    H, filtered_magnitude, edge_gaussian = gaussian_highpass_filter(image, D0)

    # 显示图像
    plt.figure(figsize=(30, 10))
    plt.subplot(1, 5, 1)
    plt.imshow(image, cmap='gray')
    plt.title('原始图像'), plt.axis('off')

    magnitude_spectrum = np.fft.fft2(image)
    plt.subplot(1, 5, 2)
    plt.imshow(np.fft.fftshift(20 * np.log(1 + np.abs(magnitude_spectrum))), cmap='gray')  # 转成中心域
    plt.title('傅里叶变换的幅度谱'), plt.axis('off')

    plt.subplot(1, 5, 3)
    plt.imshow(np.fft.fftshift(np.uint8(H * 255)), cmap='gray')
    plt.title('高斯低通滤波器'), plt.axis('off')

    plt.subplot(1, 5, 4)
    plt.imshow(np.fft.fftshift(20 * np.log(1 + np.abs(filtered_magnitude))), cmap='gray')
    plt.title('滤波后的频域幅度谱'), plt.axis('off')

    plt.subplot(1, 5, 5)
    plt.imshow(edge_gaussian, cmap='gray')
    plt.title('高斯低通滤波图像'), plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()
