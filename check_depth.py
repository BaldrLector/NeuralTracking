import cv2
import os
import numpy as np
from PIL import Image


def check_dir(root):
    color_images = [i for i in os.listdir(
        os.path.join(root, 'color')) if i.endswith('.jpg')]
    depth_images = [i for i in os.listdir(
        os.path.join(root, 'depth')) if i.endswith('.png')]
    color_images = sorted(color_images, key=lambda x: int(x.split('.')[0]))
    depth_images = sorted(depth_images, key=lambda x: int(x.split('.')[0]))
    color_images = [os.path.join(root, 'color', i) for i in color_images]
    depth_images = [os.path.join(root, 'depth', i) for i in depth_images]
    assert len(color_images) == len(depth_images)

    for depth_file, color_file in zip(depth_images, color_images):
        tgt_depth_image = cv2.imread(
            depth_file, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        tgt_depth_image = tgt_depth_image*(tgt_depth_image <= 2400)
        tgt_color_iamge = cv2.imread(color_file)
        tgt_color_iamge = tgt_color_iamge * \
            (tgt_depth_image[..., np.newaxis] > 0)
        cv2.imshow("img", (tgt_color_iamge).astype(np.uint8))
        cv2.waitKey(5)


if __name__ == '__main__':
    pass
    root = r'D:\deepdeform_v1_1\train\seq070'
    check_dir(root)
