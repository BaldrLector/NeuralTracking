from PIL import Image
import os
import numpy as np

# path = r'D:\LatexWorkshop\MM2021\Makeup\figs'
# png_files = [i for i in os.listdir(path) if i.endswith('.png')]
# for png_f in png_files:
#     image = Image.open(os.path.join(path, png_f))
#     image = np.array(image)
#     mask = image[..., -1]
#     image = image[..., :3]
#     image[mask==0] = 255
#     image = image.astype(np.uint8)
#     image = Image.fromarray(image)
#     image.convert("RGB").save(os.path.join(path, png_f[:-3]+'jpg'))

img0 = r'D:\LatexWorkshop\MM2021\Makeup\图片\0.jpg'
img1 = r'D:\LatexWorkshop\MM2021\Makeup\图片\1.jpg'
img0 = np.array(Image.open(img0))
img1 = np.array(Image.open(img1))
print(img0.shape, img1.shape)
img0[:80, -445//3+1:] = img1[:80, -445//3+1:]
img0 = Image.fromarray(img0)
img0.show()
