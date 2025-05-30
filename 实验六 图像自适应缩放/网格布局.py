import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seam_carving
from matplotlib import rcParams

# 支持汉字显示
rcParams['font.sans-serif'] = ['SimHei']


def main():
    # 读取图像
    image_path = '../resources/assets/exp6/605.png'
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 创建2x3的网格布局（2行3列）
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig)

    # 计算原始图像宽高比，用于调整子图比例
    aspect_ratio = img.shape[1] / img.shape[0]

    # 1. 原始图像（占左上角1个格子）
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img)
    ax1.set_title(f"原始图像 ({img.shape[1]}x{img.shape[0]})")
    ax1.axis('off')

    # 2. 缩小图像宽度
    target_width = img.shape[1] - 100
    reduce_width = seam_carving.resize(
        img,
        size=(target_width, img.shape[0]),
        energy_mode='backward',
        order='width-first'
    )
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(reduce_width)
    ax2.set_title(f"缩小宽度 ({reduce_width.shape[1]}x{reduce_width.shape[0]})")
    ax2.axis('off')

    # 3. 放大图像宽度
    target_width = img.shape[1] + 100
    large_wide = seam_carving.resize(
        img,
        size=(target_width, img.shape[0]),
        energy_mode='backward',
        order='width-first'
    )
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(large_wide)
    ax3.set_title(f"放大宽度 ({large_wide.shape[1]}x{large_wide.shape[0]})")
    ax3.axis('off')

    # 4. 缩小图像高度
    target_height = img.shape[0] - 50
    reduce_height = seam_carving.resize(
        img,
        size=(img.shape[1], target_height),
        energy_mode='backward',
        order='height-first'
    )
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(reduce_height)
    ax4.set_title(f"缩小高度 ({reduce_height.shape[1]}x{reduce_height.shape[0]})")
    ax4.axis('off')

    # 5. 放大图像高度
    target_height = img.shape[0] + 100
    large_height = seam_carving.resize(
        img,
        size=(img.shape[1], target_height),
        energy_mode='backward',
        order='height-first'
    )
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.imshow(large_height)
    ax5.set_title(f"放大高度 ({large_height.shape[1]}x{large_height.shape[0]})")
    ax5.axis('off')

    # 调整子图间距
    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    plt.show()


if __name__ == "__main__":
    main()
