import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from matplotlib import rcParams


rcParams['font.sans-serif'] = ['SimHei']


def laplacian_operator(img: np.ndarray,
                       kernel_type: int = 1,
                       normalize: bool = True,
                       sharpen: bool = False,
                       alpha: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    空间域拉普拉斯算子实现

    参数:
        img: 输入图像 (numpy array, 灰度图)
        kernel_type: 拉普拉斯核类型 (1=标准4邻域, 2=扩展4邻域, 3=8邻域, 4=对角线增强)
        normalize: 是否归一化输出图像 (用于显示)
        sharpen: 是否应用拉普拉斯锐化
        alpha: 锐化强度因子 (仅在sharpen=True时有效)

    返回:
        laplacian: 拉普拉斯变换结果
        sharpened: 锐化后的图像 (若sharpen=False则返回原图)
    """
    # 确保图像为float类型，避免计算时溢出
    img_float = np.float32(img)

    # 根据选择的核类型定义拉普拉斯核
    if kernel_type == 1:
        # 标准4邻域拉普拉斯核
        kernel = np.array([[0, 1, 0],
                           [1, -4, 1],
                           [0, 1, 0]], dtype=np.float32)
    elif kernel_type == 2:
        # 扩展4邻域拉普拉斯核 (权重更大)
        kernel = np.array([[0, 2, 0],
                           [2, -8, 2],
                           [0, 2, 0]], dtype=np.float32)
    elif kernel_type == 3:
        # 8邻域拉普拉斯核
        kernel = np.array([[1, 1, 1],
                           [1, -8, 1],
                           [1, 1, 1]], dtype=np.float32)
    else:
        # 对角线增强拉普拉斯核
        kernel = np.array([[2, 1, 2],
                           [1, -12, 1],
                           [2, 1, 2]], dtype=np.float32)

    # 应用卷积
    laplacian = cv2.filter2D(img_float, -1, kernel)

    print(laplacian.min(), laplacian.max())

    # 图像锐化: f_sharp = f + α(-∇²f) = f - α·∇²f
    sharpened = img_float.copy()
    if sharpen:
        sharpened = img_float - alpha * laplacian
        # 裁剪到有效范围 [0, 255]
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    # 归一化拉普拉斯结果用于显示
    if normalize:
        laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return laplacian, sharpened


def visualize_results(img: np.ndarray,
                      laplacian: np.ndarray,
                      sharpened: np.ndarray,
                      kernel_name: str):
    """可视化原始图像、拉普拉斯结果和锐化效果"""
    plt.figure(figsize=(15, 5))

    plt.subplot(131), plt.imshow(img, cmap='gray')
    plt.title('原始图像'), plt.axis('off')

    plt.subplot(132), plt.imshow(laplacian, cmap='gray')
    plt.title(f'{kernel_name} 拉普拉斯'), plt.axis('off')

    plt.subplot(133), plt.imshow(sharpened, cmap='gray')
    plt.title('拉普拉斯锐化'), plt.axis('off')

    plt.tight_layout()
    plt.show()


def main():
    # 读取图像
    img = cv2.imread('../resources/assets/exp3/308.jpg', 0)  # 以灰度模式读取

    if img is None:
        print("无法读取图像，请检查路径")
        return

    # 测试不同的拉普拉斯核
    kernel_types = {
        1: "标准4邻域",
        2: "扩展4邻域",
        3: "8邻域",
        4: "对角线增强"
    }

    for kernel_type, name in kernel_types.items():
        laplacian, sharpened = laplacian_operator(
            img,
            kernel_type=kernel_type,
            normalize=True,
            sharpen=True,
            alpha=0.5  # 锐化强度因子
        )
        visualize_results(img, laplacian, sharpened, name)


if __name__ == "__main__":
    main()
