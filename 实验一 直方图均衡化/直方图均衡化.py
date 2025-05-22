import cv2
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 支持汉字
rcParams['font.sans-serif'] = ['SimHei']


# 直方图均衡化
def histogram_equalization(image):
    # 分离图像的三个通道
    B, G, R = cv2.split(image)

    # 对每个通道进行直方图均衡化
    B_eq = cv2.equalizeHist(B)
    G_eq = cv2.equalizeHist(G)
    R_eq = cv2.equalizeHist(R)

    # 合并均衡化后的通道
    image_eq = cv2.merge((B_eq, G_eq, R_eq))
    return image_eq


# 显示图像
def show_image(image, image_eq):
    # 图像信息
    img_info = [
        {'index': 1, 'title': '原始图像', 'image': image},
        {'index': 2, 'title': '直方图均衡化的图像', 'image': image_eq},
    ]
    # 创建子图
    for item in img_info:
        plt.subplot(2, 2, item['index'])
        plt.title(item['title'])
        plt.imshow(item['image'], cmap='gray')
        plt.axis('off')


# 显示直方图
def show_histogram(image, image_eq):
    # 定义颜色
    colors = ('b', 'g', 'r')

    # 直方图信息
    histogram_info = [
        {'index': 3, 'title': '原始直方图', 'image': image},
        {'index': 4, 'title': '直方图均衡化的直方图', 'image': image_eq},
    ]
    # 创建子图
    for item in histogram_info:
        plt.subplot(2, 2, item['index'])
        plt.title(item['title'])
        plt.xlabel('像素值')
        plt.ylabel('频率')
        for i, color in enumerate(colors):
            hist = cv2.calcHist(item['image'], [i], None, [256], [0, 256])
            plt.plot(hist, color=color)


# 主函数
def main():
    # 获取图像路径
    image_path = '../resources/assets/exp1/106.jpg'

    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        exit(1)

    # 转换颜色空间从 BGR 到 RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 直方图均衡化
    image_eq = histogram_equalization(image)

    # 显示原始图像和均衡化后的图像
    plt.figure(figsize=(12, 6))  # 大小为 12 * 6
    show_image(image, image_eq)

    # 绘制原始图像的直方图
    show_histogram(image, image_eq)

    # 调整子图之间的间距
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
