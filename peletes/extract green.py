import cv2
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


img = cv2.imread('img/tar_ver.jpg')
ds_factor = 0.1
rgb_image = cv2.resize(img, (int(img.shape[1] * ds_factor),int(img.shape[0] * ds_factor)), interpolation = cv2.INTER_AREA)
hsv_nemo = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
nemo = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
h, s, v = cv2.split(hsv_nemo)
fig = plt.figure()

axis = fig.add_subplot(1, 1, 1, projection='3d')
pixel_colors = nemo.reshape((np.shape(nemo)[0]*np.shape(nemo)[1], 3))
norm = colors.Normalize(vmin=-1.,vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()
axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Hue")
axis.set_ylabel("Saturation")
axis.set_zlabel("Value")
plt.show()