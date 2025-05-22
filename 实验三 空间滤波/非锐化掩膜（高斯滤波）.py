import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 支持汉字
rcParams['font.sans-serif'] = ['SimHei']


# 非锐化掩膜
def un_sharp_mask(img, kernel_size=(5, 5), alpha=1.6, beta=-0.5):
    """
    应用非锐化掩模进行图像增强

    :param img: 输入图像（BGR 格式）
    :param kernel_size: 高斯滤波核大小，默认为 (5, 5)
    :param alpha: 原始图像的权重，默认为 1.5
    :param beta: 高斯滤波图像的权重，默认为 -0.5
    :return: 增强后的图像
    """
    # 高斯滤波平滑
    gaussian_blurred = cv2.GaussianBlur(img, kernel_size, 0)
    # 非锐化掩膜
    un_sharp_masked = cv2.addWeighted(img, alpha, gaussian_blurred, beta, 0)
    # 确保非锐化掩模结果在有效范围内
    un_sharp_masked = np.clip(un_sharp_masked, 0, 255).astype(np.uint8)
    return gaussian_blurred, un_sharp_masked


# 主函数
def main():
    # 读取图像
    image_path = '../resources/assets/exp3/310.png'
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        exit(1)

    # 应用非锐化掩模
    gaussian_image, enhanced_image = un_sharp_mask(image)

    # 图像信息
    img_info = [
        {'index': 1, 'title': '原始图像', 'image': image},
        {'index': 2, 'title': '高斯滤波图像', 'image': gaussian_image},
        {'index': 3, 'title': '非锐化掩膜图像', 'image': enhanced_image},
    ]

    # 原图像
    plt.figure(figsize=(12, 6))
    for item in img_info:
        plt.subplot(1, len(img_info), item['index'])
        plt.title(item['title'])
        plt.imshow(item['image'], cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
