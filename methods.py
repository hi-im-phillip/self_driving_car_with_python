import pyautogui
import cv2
import numpy as np

path_dir_ss = "E:\\self_driving_car_SS\\snapshot.png"


def take_ss():
    '''
    Take a screenshot of the screen and store it in memory, then
    convert PIL/Pillow image to an OpenCV compatible Numpy array
    and write the image to disk
    '''

    image = pyautogui.screenshot()
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    cv2.imwrite(path_dir_ss, image)
    #return image


take_ss()