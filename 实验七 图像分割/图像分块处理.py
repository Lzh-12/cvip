import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import data
from matplotlib import rcParams

# 支持汉字
rcParams['font.sans-serif'] = ['SimHei']


def otsu_threshold(image):
    """
    使用Otsu方法计算最佳阈值

    参数:
    image: 输入的单通道灰度图像

    返回:
    threshold: 最佳阈值
    binary_image: 二值化后的图像
    hist: 图像直方图
    """
    # 确保图像是单通道灰度图
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 计算直方图
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_norm = hist.ravel() / hist.sum()

    # 计算累积分布函数和均值
    cdf = hist_norm.cumsum()
    mean = np.sum(np.arange(256) * hist_norm)

    # 初始化最大类间方差和最佳阈值
    max_variance = 0
    threshold = 0

    # 遍历所有可能的阈值(0-255)
    for t in range(256):
        # 计算前景和背景的概率
        w0 = cdf[t]
        w1 = 1 - w0

        if w0 == 0 or w1 == 0:
            continue

        # 计算前景和背景的均值
        mean0 = np.sum(np.arange(t + 1) * hist_norm[:t + 1]) / w0
        mean1 = np.sum(np.arange(t + 1, 256) * hist_norm[t + 1:]) / w1

        # 计算类间方差
        variance = w0 * w1 * (mean0 - mean1) ** 2

        # 更新最大方差和阈值
        if variance > max_variance:
            max_variance = variance
            threshold = t

    # 应用阈值进行二值化
    _, binary_image = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    return threshold, binary_image, hist


def block_otsu_threshold(image, num_rows=None, num_cols=None):
    """
    将图像分块并对每块应用Otsu阈值处理

    参数:
    image: 输入的单通道灰度图像
    num_rows: 垂直方向分块数（若未指定，根据图像高度自适应）
    num_cols: 水平方向分块数（若未指定，根据图像宽度自适应）

    返回:
    binary_image_blocks: 分块处理后的二值化图像列表
    thresholds: 每个子块的最佳阈值列表
    """
    height, width = image.shape
    if num_rows is None:
        num_rows = int(np.sqrt(height // 30))  # 示例：根据高度自适应，可调整
    if num_cols is None:
        num_cols = int(np.sqrt(width // 30))  # 示例：根据宽度自适应，可调整

    block_height = height // num_rows
    block_width = width // num_cols

    binary_image_blocks = []
    thresholds = []
    for r in range(num_rows):
        for c in range(num_cols):
            block = image[r * block_height: (r + 1) * block_height, c * block_width: (c + 1) * block_width]
            threshold, binary_block, _ = otsu_threshold(block)
            binary_image_blocks.append(binary_block)
            thresholds.append(threshold)

    return binary_image_blocks, thresholds


def morphological_postprocess(image):
    """
    对图像进行形态学后处理，先膨胀后腐蚀，去除噪点并填充空洞
    """
    kernel = np.ones((3, 3), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.erode(image, kernel, iterations=1)
    return image


def apply_filter(image, filter_type='gaussian', kernel_size=(5, 5), sigma=5.0):
    """
    应用滤波算法

    参数:
    image: 输入图像
    filter_type: 滤波算法类型，可选'gaussian'（高斯滤波）、'median'（中值滤波）等
    kernel_size: 滤波核大小
    sigma: 高斯滤波标准差（仅在高斯滤波时有效）

    返回:
    filtered_image: 滤波后的图像
    """
    if filter_type == 'gaussian':
        return cv2.GaussianBlur(image, kernel_size, sigma)
    elif filter_type =='median':
        return cv2.medianBlur(image, kernel_size[0])  # 中值滤波核大小应为奇数
    else:
        return image


def main():
    # 图像路径
    image_path = '../resources/assets/exp7/704.png'
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print("无法加载图像，请检查路径!")
        from skimage import data
        image = np.array(data.camera())
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        print("使用示例图像进行演示")

    # 转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 高斯滤波
    kernel_size = (5, 5)  # 高斯核大小
    sigma = 5.0  # 标准差
    blurred_image = apply_filter(gray_image, filter_type='gaussian', kernel_size=kernel_size, sigma=sigma)

    # 全局Otsu阈值分割
    _, global_binary_image, global_hist = otsu_threshold(gray_image)
    _, blurred_global_binary_image, blurred_global_hist = otsu_threshold(blurred_image)

    # 分块Otsu阈值分割
    binary_image_blocks, thresholds = block_otsu_threshold(gray_image)
    blurred_binary_image_blocks, blurred_thresholds = block_otsu_threshold(blurred_image)

    # 重新拼接分块后的图像
    num_rows = len(binary_image_blocks) // len(binary_image_blocks[0])
    num_cols = len(binary_image_blocks[0])
    result_image = np.vstack([np.hstack(binary_image_blocks[i * num_cols: (i + 1) * num_cols]) for i in range(num_rows)])
    blurred_result_image = np.vstack([np.hstack(blurred_binary_image_blocks[i * num_cols: (i + 1) * num_cols]) for i in range(num_rows)])

    # 形态学后处理
    result_image = morphological_postprocess(result_image)
    blurred_result_image = morphological_postprocess(blurred_result_image)

    # 显示结果
    plt.figure(figsize=(18, 10))

    # 原始图像
    plt.subplot(241)
    plt.title('原始图像')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cmap='gray')
    plt.axis('off')

    # 高斯滤波后图像
    plt.subplot(245)
    plt.title(f'高斯滤波后 (核大小={kernel_size}, σ={sigma})')
    plt.imshow(blurred_image, cmap='gray')
    plt.axis('off')

    # 原始图像的全局Otsu分割
    plt.subplot(242)
    plt.title('原始图像全局Otsu分割')
    plt.imshow(global_binary_image, cmap='gray')
    plt.axis('off')

    # 滤波后图像的全局Otsu分割
    plt.subplot(246)
    plt.title('滤波后图像全局Otsu分割')
    plt.imshow(blurred_global_binary_image, cmap='gray')
    plt.axis('off')

    # 原始图像的分块Otsu分割
    plt.subplot(243)
    plt.title('原始图像分块Otsu分割')
    plt.imshow(result_image, cmap='gray')
    plt.axis('off')

    # 滤波后图像的分块Otsu分割
    plt.subplot(247)
    plt.title('滤波后图像分块Otsu分割')
    plt.imshow(blurred_result_image, cmap='gray')
    plt.axis('off')

    # 原始图像直方图
    plt.subplot(244)
    plt.title('原始图像直方图')
    plt.plot(global_hist, color='black')
    for t in thresholds:
        plt.axvline(x=t, color='r', linestyle='--', label=f'子块阈值')
    plt.xlabel('像素值')
    plt.ylabel('像素数量')
    plt.legend()

    # 滤波后图像直方图
    plt.subplot(248)
    plt.title('滤波后图像直方图')
    plt.plot(blurred_global_hist, color='black')
    for t in blurred_thresholds:
        plt.axvline(x=t, color='g', linestyle='--', label=f'子块阈值')
    plt.xlabel('像素值')
    plt.ylabel('像素数量')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()