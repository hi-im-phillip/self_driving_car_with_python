import cv2
import numpy as np

screen_height = 800
screen_height = 610


def pre_process_frame(frame):
    """
    :param frame then convert it Gray, from there
    blur the given image and apply Canny effect
    :return: Return it with made changes
    """
    processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    processed_frame = cv2.Canny(processed_frame, 200, 350)
    processed_frame = cv2.GaussianBlur(processed_frame, (5, 5), 0)
    processed_frame = region_of_interest(frame=processed_frame)
    return processed_frame


def region_of_interest(frame):
    """
    :param
    Takes a image and return enclosed region of field of view,
    and enclosed region is triangle shape
    triangle shape - 200, bottom - 700, bottom - 400, 200
    mask - create same shape of black mask as give frame
    fillPolly - take a triangle(polygons) which borders are defined and apply it on mask and that area will be white
    :return: masked_frame - computing the bitwise & of both images, ultimately masking the canny image to only show the
    region of interest traced by the polygonal contour of the mask
    """
    mask = np.zeros_like(frame)
    vertices = np.array([[[0, 600], [0, 400], [200, 320], [600, 320], [screen_height, 400], [screen_height, 600]]]) #np.array([[[50, 475], [0, 600], [0, 350], [200, 300], [600, 300], [800, 350], [800, 600], [750, 475]]]) # np.array([[(100, height), (700, height), (400, 100)]])
    cv2.fillPoly(mask, vertices, 255)
    masked_frame = cv2.bitwise_and(frame, mask)
    return masked_frame
