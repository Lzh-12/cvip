import cv2
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 支持汉字
rcParams['font.sans-serif'] = ['SimHei']


# Laplacian算子
def apply_laplacian(image):
    """
    laplacian算子处理
    :param image:
    :return: 处理后的图像
    """
    # 使用CV_16S避免计算过程中溢出
    laplacian = cv2.Laplacian(image, cv2.CV_16S, ksize=3, scale=1, delta=0)
    # 转换回8位无符号整数
    laplacian_abs = cv2.convertScaleAbs(laplacian, alpha=0.2, beta=0)

    # 图像增强: 原始图像 + 拉普拉斯结果
    enhanced = cv2.add(image, laplacian_abs)

    return enhanced, laplacian_abs


# 主函数
def main():
    # 获取图像路径（相对当前脚本目录）
    image_path = '../resources/assets/exp3/308.jpg'

    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        exit(1)

    # 转换为灰度图像
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 拉普拉斯算子
    enhanced, laplacian_result = apply_laplacian(image)

    # 显示图像
    img_info = [
        {'index': 1, 'title': '原始图像', 'image': image},
        {'index': 2, 'title': '拉普拉斯算子结果', 'image': laplacian_result},
        {'index': 3, 'title': '增强后的图像', 'image': enhanced},
    ]

    # 创建子图
    plt.figure(figsize=(15, 5))
    for item in img_info:
        plt.subplot(1, len(img_info), item['index'])
        plt.title(item['title'])
        plt.imshow(item['image'], cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
