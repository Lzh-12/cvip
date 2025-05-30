import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import rcParams

# 支持汉字
rcParams['font.sans-serif'] = ['SimHei']


class SeamCarver:
    def __init__(self, image_path):
        """初始化Seam Carving处理器"""
        self.image = cv2.imread(image_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.original_image = self.image.copy()
        self.height, self.width = self.image.shape[:2]

    def compute_energy(self, img=None):
        """计算图像能量图（使用Sobel算子）"""
        if img is None:
            img = self.image

        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 计算x和y方向的梯度
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # 计算梯度幅值作为能量
        energy = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        return energy

    def find_vertical_seam(self, energy):
        """寻找垂直方向的最低能量seam"""
        height, width = energy.shape
        dp = energy.copy()  # 动态规划表

        # 构建动态规划表
        for i in range(1, height):
            for j in range(width):
                if j == 0:
                    min_prev = min(dp[i - 1, j], dp[i - 1, j + 1])
                elif j == width - 1:
                    min_prev = min(dp[i - 1, j - 1], dp[i - 1, j])
                else:
                    min_prev = min(dp[i - 1, j - 1], dp[i - 1, j], dp[i - 1, j + 1])
                dp[i, j] += min_prev

        # 回溯找到路径
        seam = np.zeros(height, dtype=int)
        seam[-1] = np.argmin(dp[-1])

        for i in range(height - 2, -1, -1):
            j = seam[i + 1]
            if j == 0:
                candidates = [j, j + 1]
            elif j == width - 1:
                candidates = [j - 1, j]
            else:
                candidates = [j - 1, j, j + 1]
            seam[i] = candidates[np.argmin(dp[i, candidates])]

        return seam

    def find_horizontal_seam(self, energy):
        """寻找水平方向的最低能量seam（通过转置实现）"""
        # 转置图像和能量图
        transposed_energy = energy.T
        seam = self.find_vertical_seam(transposed_energy)
        return seam

    def remove_vertical_seam(self, seam):
        """移除垂直seam"""
        height, width = self.image.shape[:2]
        new_image = np.zeros((height, width - 1, 3), dtype=np.uint8)

        for i in range(height):
            j = seam[i]
            # 复制seam左侧的像素
            if j > 0:
                new_image[i, :j] = self.image[i, :j]
            # 复制seam右侧的像素
            if j < width - 1:
                new_image[i, j:] = self.image[i, j + 1:]

        self.image = new_image
        self.width -= 1
        return new_image

    def remove_horizontal_seam(self, seam):
        """移除水平seam（通过转置实现）"""
        # 转置图像
        self.image = np.transpose(self.image, (1, 0, 2))
        self.height, self.width = self.width, self.height

        # 移除垂直seam
        new_image = self.remove_vertical_seam(seam)

        # 转回原方向
        self.image = np.transpose(new_image, (1, 0, 2))
        self.height, self.width = self.width, self.height
        return self.image

    def resize(self, new_width=None, new_height=None, show_animation=False):
        """调整图像尺寸"""
        if new_width is None and new_height is None:
            raise ValueError("至少指定一个新的宽度或高度")

        # 记录原始图像用于动画
        frames = [self.image.copy()]

        # 调整宽度（垂直seam）
        if new_width is not None and new_width != self.width:
            delta_width = self.width - new_width
            if delta_width > 0:  # 需要减小宽度
                for i in range(delta_width):
                    energy = self.compute_energy()
                    seam = self.find_vertical_seam(energy)
                    self.remove_vertical_seam(seam)
                    if show_animation and i % 5 == 0:  # 每5次记录一帧
                        frames.append(self.image.copy())
            else:  # 需要增加宽度（seam insertion）
                # 这里简化处理，实际实现需要更复杂的逻辑
                pass

        # 调整高度（水平seam）
        if new_height is not None and new_height != self.height:
            delta_height = self.height - new_height
            if delta_height > 0:  # 需要减小高度
                for i in range(delta_height):
                    energy = self.compute_energy()
                    seam = self.find_horizontal_seam(energy)
                    self.remove_horizontal_seam(seam)
                    if show_animation and i % 5 == 0:  # 每5次记录一帧
                        frames.append(self.image.copy())
            else:  # 需要增加高度
                # 这里简化处理，实际实现需要更复杂的逻辑
                pass

        if show_animation:
            return self.image, frames
        else:
            return self.image

    def visualize_seam(self, seam, is_vertical=True):
        """可视化seam路径"""
        marked_image = self.image.copy()
        height, width = self.image.shape[:2]

        if is_vertical:
            for i in range(height):
                j = seam[i]
                marked_image[i, j] = [255, 0, 0]  # 红色标记
        else:
            for j in range(width):
                i = seam[j]
                marked_image[i, j] = [255, 0, 0]  # 红色标记

        return marked_image


def main():
    """主函数，演示seam carving功能"""
    image_path = '../resources/assets/exp6/606.png'
    carver = SeamCarver(image_path)

    # 调整图像宽度（例如减小50像素）
    new_width = carver.width - 100
    new_image, frames = carver.resize(new_width=new_width, show_animation=True)

    # 显示原始图像和调整后的图像
    plt.figure(figsize=(12, 6))

    plt.subplot(121)
    plt.title('原始图像 ({0}x{1})'.format(carver.original_image.shape[1], carver.original_image.shape[0]))
    plt.imshow(carver.original_image, aspect='equal')
    plt.axis('off')

    plt.subplot(122)
    plt.title('调整后图像 ({0}x{1})'.format(new_image.shape[1], new_image.shape[0]))
    plt.imshow(new_image, aspect='equal')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # 可选：创建seam carving过程的动画
    if frames and len(frames) > 1:
        fig = plt.figure()
        ims = []
        for frame in frames:
            im = plt.imshow(frame, animated=True)
            ims.append([im])

        ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True)
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    main()
