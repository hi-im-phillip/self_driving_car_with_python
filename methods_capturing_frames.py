import cv2
import numpy as np
import time
from self_driving_car_with_python.grab_screen import grab_screen


def region_of_interest(frame):
    '''
      Takes a image and return enclosed region of field of view,
      and enclosed region is triangle shape
      height - height of the given frame, x-cord[0]
      triangle shape - 200, bottom - 700, bottom - 400, 200
      mask - create same shape of black mask as give frame
      fillPolly - take a triangle(polygons) which borders are defined and apply it on mask and that area will be white
     '''
    height = frame.shape[0]
    polygons = np.array([[(200, height), (700, height), (400, 200)]])
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, polygons, 255)
    return mask


def make_canny_frame(frame):
    '''
    Take a frame and then convert it Gray, from there
    blur the given image and apply Canny effect
    Return it with made changes
    '''
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    canny_frame = cv2.Canny(blur_frame, 50, 150)
    return canny_frame


def make_frames():
    '''
       Take a frame of the screen and store it in memory
       region - Specifies specific region (bbox = x cord, y cord, width, height)
       '''
    while True:
        time_test = time.time()
        frame = grab_screen(region=(0, 0, 800, 640))
        canny_frame = make_canny_frame(frame=frame)
        cv2.imshow("window", region_of_interest(canny_frame))
        print(time.time() - time_test)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            cv2.destroyAllWindows()
            break

