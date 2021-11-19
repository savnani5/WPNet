import cv2
from PIL import Image
import numpy as np
from skimage import io, transform
import matplotlib.pyplot as plt

file = "my_examples\img__0_1549120949212458600.png"
# im = np.clip(np.asarray(Image.open(file), dtype=float) / 255, 0, 1)
# im = np.asarray(Image.open(file), dtype=float)

# img = cv2.imread(file)
# img = cv2.cvtColor(img,cv2.COLOR_RGBA2RGB)
# img = np.clip(np.asarray(img, dtype=float)/255, 0, 1)

img = io.imread(file)

print(type(img), img.dtype, img.shape)

# print(type(im), im.dtype, im.shape)

io.imshow(img)
plt.show()

# cv2.imshow("forest", img)
# cv2.waitKey(0)

# im.show()


