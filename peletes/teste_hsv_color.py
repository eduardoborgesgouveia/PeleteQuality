from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt
import numpy as np

low_green = np.array([71, 255, 255])
#high_green = np.array([94, 255, 94])
high_green = np.array([101, 255, 80])

lo_square = np.full((10, 10, 3), low_green, dtype=np.uint8) / 255.0
do_square = np.full((10, 10, 3), high_green, dtype=np.uint8) / 255.0

plt.subplot(1, 2, 1)
plt.imshow(hsv_to_rgb(do_square))
plt.subplot(1, 2, 2)
plt.imshow(hsv_to_rgb(lo_square))
plt.show()