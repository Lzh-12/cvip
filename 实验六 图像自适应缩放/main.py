import cv2
from matplotlib import pyplot as plt

from seam_carving import SeamCarver
from matplotlib import rcParams

# 支持汉字
rcParams['font.sans-serif'] = ['SimHei']


def image_resize_without_mask(filename_input, filename_output, new_height, new_width):
    obj = SeamCarver(filename_input, new_height, new_width)
    obj.save_result(filename_output)


def image_resize_with_mask(filename_input, filename_output, new_height, new_width, filename_mask):
    obj = SeamCarver(filename_input, new_height, new_width, protect_mask=filename_mask)
    obj.save_result(filename_output)


def object_removal(filename_input, filename_output, filename_mask):
    obj = SeamCarver(filename_input, 0, 0, object_mask=filename_mask)
    obj.save_result(filename_output)


if __name__ == '__main__':
    """
    Put image in in/images folder and protect or object mask in in/masks folder
    Ouput image will be saved to out/images folder with filename_output
    """

    folder_in = 'in'
    folder_out = 'out'

    filename_input = '../resources/assets/exp6/606.png'
    filename_output = 'image_result.png'
    filename_mask = 'mask.jpg'
    new_height = 308
    new_width = 300

    input_image = filename_input
    output_image = filename_output
    input_mask = filename_mask

    # input_image = os.path.join(folder_in, "images", filename_input)
    # input_mask = os.path.join(folder_in, "masks", filename_mask)
    # output_image = os.path.join(folder_out, "images", filename_output)

    image_resize_without_mask(input_image, output_image, new_height, new_width)
    # image_resize_with_mask(input_image, output_image, new_height, new_width, input_mask)
    # object_removal(input_image, output_image, input_mask)

    image = cv2.imread(filename_input)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.title('原始图像 ({0}x{1})'.format(image.shape[1], image.shape[0]))
    plt.imshow(image, aspect='equal')
    plt.axis('off')

    new_image = cv2.imread(filename_output)
    new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    plt.subplot(122)
    plt.title('原始图像 ({0}x{1})'.format(new_image.shape[1], new_image.shape[0]))
    plt.imshow(new_image, aspect='equal')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
