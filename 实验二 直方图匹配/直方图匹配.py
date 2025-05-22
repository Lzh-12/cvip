import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 支持汉字
rcParams['font.sans-serif'] = ['SimHei']


# 直方图匹配
def histogram_matching(source_img, target_img):
    # 计算源图像和目标图像的直方图
    source_hist, bins1 = np.histogram(source_img, bins=256, range=[0, 256])
    target_hist, bins2 = np.histogram(target_img, bins=256, range=[0, 256])

    # 计算累积直方图
    source_cdf = source_hist.cumsum()
    source_cdf = (source_cdf / source_cdf[-1]).astype(np.float32)

    target_cdf = target_hist.cumsum()
    target_cdf = (target_cdf / target_cdf[-1]).astype(np.float32)

    # 使用累积直方图进行直方图匹配
    matched_cdf = np.interp(source_cdf, target_cdf, range(256)).astype(np.uint8)
    matched_img = matched_cdf[source_img]
    return matched_img


# 主函数
def main():
    # 获取图像
    source_path = '../resources/assets/exp2/206.jpg'
    target_path = '../resources/assets/exp2/209.jpg'

    # 读取源图像和目标图像
    source_img = cv2.imread(source_path)
    target_img = cv2.imread(target_path)
    if source_img is None or target_img is None:
        print(f"无法读取图像: {source_path} 或 {target_path}")
        exit(1)

    # 转换为BGR格式
    source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
    target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)

    # 进行直方图匹配
    matched_img = histogram_matching(source_img, target_img)

    # 显示结果
    plt.figure(figsize=(12, 6))  # 大小为 12 * 6

    # 图像信息
    img_info = [
        {'index': 1, 'title': '原始图像', 'image': source_img},  # 原始图像
        {'index': 2, 'title': '参考图像', 'image': target_img},  # 参考图像
        {'index': 3, 'title': '匹配图像', 'image': matched_img},  # 匹配图像
    ]

    # 创建子图
    for item in img_info:
        plt.subplot(2, 3, item['index'])
        plt.title(item['title'])
        plt.imshow(item['image'], cmap='gray')
        plt.axis('off')

    # 直方图信息
    histogram_info = [
        {'index': 4, 'title': '原始直方图', 'image': source_img},
        {'index': 5, 'title': '参考直方图', 'image': target_img},
        {'index': 6, 'title': '匹配直方图', 'image': matched_img},
    ]

    # 定义颜色
    colors = ('b', 'g', 'r')
    # 创建子图
    for item in histogram_info:
        plt.subplot(2, 3, item['index'])
        plt.title(item['title'])
        plt.xlabel('像素值')
        plt.ylabel('频率')
        for i, color in enumerate(colors):
            hist = cv2.calcHist(item['image'], [i], None, [256], [0, 256])
            plt.plot(hist, color=color)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
