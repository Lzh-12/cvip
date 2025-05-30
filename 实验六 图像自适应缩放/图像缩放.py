import cv2
import matplotlib.pyplot as plt
import seam_carving
from matplotlib import rcParams

# 支持汉字
rcParams['font.sans-serif'] = ['SimHei']


def main():
    # 读取图像
    image_path = '../resources/assets/exp6/606.png'
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 显示原始图像
    plt.figure(figsize=(15, 10))
    plt.subplot(231)
    plt.imshow(img)
    plt.title(f"原始图像 ({img.shape[1]}x{img.shape[0]})")
    plt.axis('off')

    # 1：缩小图像宽度（从原始宽度到目标宽度）
    target_width = img.shape[1] - 100  # 减少100像素宽度
    reduce_with = seam_carving.resize(
        img,
        size=(target_width, img.shape[0]),  # 保持高度不变
        energy_mode='backward',  # 能量模式：'backward'（默认）或'forward'
        order='width-first'  # 调整顺序：先处理宽度
    )

    # 2：放大图像宽度（从原始宽度到目标宽度）
    target_width = img.shape[1] + 100  # 增加100像素宽度
    large_wide = seam_carving.resize(
        img,
        size=(target_width, img.shape[0]),  # 保持高度不变
        energy_mode='backward',
        order='width-first'
    )

    # 3：缩小图像高度（从原始高度到目标高度）
    target_height = img.shape[0] - 100  # 减少100像素高度
    reduce_height = seam_carving.resize(
        img,
        size=(img.shape[1], target_height),  # 保持宽度不变
        energy_mode='backward',  # 能量模式：'backward'（默认）或'forward'
        order='height-first'  # 调整顺序：先处理高度
    )

    # 4：放大图像高度（从原始高度到目标高度）
    target_height = img.shape[0] + 100  # 减少100像素高度
    large_height = seam_carving.resize(
        img,
        size=(img.shape[1], target_height),  # 保持宽度不变
        energy_mode='backward',  # 能量模式：'backward'（默认）或'forward'
        order='height-first'  # 调整顺序：先处理高度
    )

    # 显示结果
    plt.subplot(232)
    plt.imshow(reduce_with)
    plt.title(f"缩小宽度 ({reduce_with.shape[1]}x{reduce_with.shape[0]})")
    plt.axis('off')

    plt.subplot(333)
    plt.imshow(large_wide)
    plt.title(f"放大宽度 ({large_wide.shape[1]}x{large_wide.shape[0]})")
    plt.axis('off')

    plt.subplot(235)
    plt.imshow(reduce_height)
    plt.title(f"缩小高度 ({reduce_height.shape[1]}x{reduce_height.shape[0]})")
    plt.axis('off')

    plt.subplot(236)
    plt.imshow(large_height)
    plt.title(f"放大高度 ({large_height.shape[1]}x{large_height.shape[0]})")
    plt.axis('off')

    # plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
