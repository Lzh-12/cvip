import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 支持汉字
rcParams['font.sans-serif'] = ['SimHei']


# Sobel 算子滤波器
def apply_sobel(image):
    # 计算Sobel卷积结果 （输出图像的深度为 16 位有符号整数， 表示增强边缘检测效果
    # x方向的Sobel卷积结果
    x = cv.Sobel(image, cv.CV_16S, 1, 0, ksize=-1)
    # 计算y方向的Sobel卷积结果
    y = cv.Sobel(image, cv.CV_16S, 0, 1, ksize=-1)

    # 将数据进行转换
    Scale_absX = cv.convertScaleAbs(x)
    Scale_absY = cv.convertScaleAbs(y)

    # 结果合成
    result = cv.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)
    return result


# 主函数
def main():
    # 读取图像
    img_path = '../resources/assets/exp3/314.jpg'
    # 读取灰度图像
    image = cv.imread(img_path)
    if image is None:
        print(f"无法读取图像: {img_path}")
        exit(1)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    # 转成为灰度图像
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Sobel算子滤波
    sobel_image = apply_sobel(gray_image)

    # 图像显示
    plt.figure(figsize=(10, 8))
    plt.subplot(1, 2, 1)
    plt.title('原始图像')
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Sobel滤波图像')
    plt.imshow(sobel_image, cmap='gray')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()
