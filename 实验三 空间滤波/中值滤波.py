import cv2
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 支持汉字
rcParams['font.sans-serif'] = ['SimHei']

# 读取图像
image = cv2.imread('../resources/assets/exp3/304.jpg')
if image is None:
    print("无法读取图像")
else:
    # 添加椒盐噪声
    # noisy_image = cv2.randu(image.shape, 0, 256)
    # noisy_image[noisy_image > 200] = 255
    # noisy_image[noisy_image < 50] = 0

    # 中值滤波去除噪声
    # median_filtered_image = cv2.medianBlur(noisy_image, 5)
    median_filtered_image = cv2.medianBlur(image, 5)

    # 显示结果
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.title('原始图像(椒盐噪声)')
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    # plt.subplot(1, 3, 2)
    # plt.title('Noisy Image')
    # plt.imshow(noisy_image, cmap='gray')
    # plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('中值滤波后的图像')
    plt.imshow(median_filtered_image, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
