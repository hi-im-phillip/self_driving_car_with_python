# https://stackoverflow.com/questions/35097837/capture-video-data-from-screen-in-python

import cv2
import numpy as np
from PIL import ImageGrab
import time

path_dir_ss = "E:\\self_driving_car_SS\\snapshot.png"


def make_frames():
    '''
       Take a screenshot of the screen and store it in memory, then
       convert PIL/Pillow image to an OpenCV compatible Numpy array
       and show it on screen
       bbox - Specifies specific region (bbox = x cord, y cord, width, height)
       '''
    while True:
        time_test = time.time()
        img = ImageGrab.grab(bbox=(0, 0, 800, 640))
        img_np = np.array(img)
        frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        cv2.imshow("test", frame)
        print(time.time() - time_test)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            cv2.destroyAllWindows()
            break


make_frames()