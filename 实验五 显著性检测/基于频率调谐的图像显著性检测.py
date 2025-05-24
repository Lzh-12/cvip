import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams

# 支持汉字
rcParams['font.sans-serif'] = ['SimHei']


# 基于频率调谐的图像显著性检测
def frequency_tuned_saliency():
    # 图像路径
    image_path = '../resources/assets/exp5/501.png'
    # 读取图像
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 步骤（1）：输入图像进行高斯滤波，去除高频信息
    gaussian_blur_image = cv2.GaussianBlur(image_rgb, (15, 15), 0)

    # 创建3x2的图像网格
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.title('原始图像')
    plt.imshow(image_rgb)
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title('高斯滤波后图像')
    plt.imshow(gaussian_blur_image)
    plt.axis('off')

    # 步骤（2）：将原图与滤波后的图像从RGB颜色空间转换到Lab颜色空间
    lab_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    lab_blur_image = cv2.cvtColor(gaussian_blur_image, cv2.COLOR_RGB2LAB)
    # plt.subplot(3, 2, 3)
    # plt.title('原始图像Lab空间')
    # plt.imshow(cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB))
    # plt.axis('off')
    #
    # plt.subplot(3, 2, 4)
    # plt.title('高斯滤波后图像Lab空间')
    # plt.imshow(cv2.cvtColor(lab_blur_image, cv2.COLOR_LAB2RGB))
    # plt.axis('off')

    # 步骤（3）：在原图上计算Lab空间颜色向量平均值
    L_mean = np.mean(lab_image[:, :, 0])  # 亮度通道
    a_mean = np.mean(lab_image[:, :, 1])  # 红绿通道
    b_mean = np.mean(lab_image[:, :, 2])  # 蓝黄通道
    I_mu = np.array([L_mean, a_mean, b_mean])

    # 创建平均颜色图像
    mean_color_image = np.zeros_like(lab_image, dtype=np.float32)
    mean_color_image[:, :, 0] = L_mean
    mean_color_image[:, :, 1] = a_mean
    mean_color_image[:, :, 2] = b_mean
    mean_color_image_rgb = cv2.cvtColor(mean_color_image.astype(np.uint8), cv2.COLOR_LAB2RGB)

    plt.subplot(2, 2, 3)
    plt.title('平均颜色图像')
    plt.imshow(mean_color_image_rgb)
    plt.axis('off')

    # 步骤（4）：计算三个通道均值与高斯滤波后图像的欧氏距离之和
    height, width, _ = lab_blur_image.shape
    saliency_map = np.zeros((height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            I_whc = lab_blur_image[y, x]  # 获取当前像素点的颜色向量
            distance = np.linalg.norm(I_mu - I_whc)  # 计算颜色向量的欧氏距离
            saliency_map[y, x] = distance

    # 步骤（5）：对S做归一化得到显著度图
    normalized_saliency_map = cv2.normalize(saliency_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    plt.subplot(2, 2, 4)
    plt.title('归一化显著图')
    plt.imshow(normalized_saliency_map, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    frequency_tuned_saliency()
