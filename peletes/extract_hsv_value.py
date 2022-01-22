import cv2 
import numpy as np
greenBGR = np.uint8([[[80,52,0 ]]])
 
hsv_green = cv2.cvtColor(greenBGR,cv2.COLOR_BGR2HSV)
print (hsv_green)