import cv2
import numpy as np

black_image = np.zeros((512,512), np.uint8)
white_image = np.ones((512,512), np.uint8) * 255
grey_image = np.ones((512,512), np.uint8) * 127

for i in range(1, 500):
    b_name = f'black_field_{i:03d}.png' 
    w_name = f'white_field_{i:03d}.png' 
    g_name = f'grey_field__{i:03d}.png' 
    cv2.imwrite(b_name, black_image)
    cv2.imwrite(w_name, white_image)
    cv2.imwrite(g_name, grey_image)

