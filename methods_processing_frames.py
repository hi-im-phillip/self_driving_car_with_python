import cv2
import numpy as np

screen_height = 800
screen_height = 640


def pre_process_frame(frame):
    """
    :param frame then convert it Gray, from there
    blur the given image and apply Canny effect
    :return: Return it with made changes
    """
    vertices = np.array([[10, 490], [10, 370], [200, 280], [600, 280], [800, 370], [800, 490]], np.int32)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # gray image
    canny_effect_frame = cv2.Canny(gray_frame, 100, 200)  # applied canny effect
    blur_frame = cv2.GaussianBlur(canny_effect_frame, (5, 5), 0)
    frame_with_roi = region_of_interest(blur_frame, [vertices])
    lines = cv2.HoughLinesP(frame_with_roi,
                            rho=1,
                            theta=np.pi/180,
                            threshold=180,
                            lines=np.array([]),
                            minLineLength=30,
                            maxLineGap=15)
    return frame_with_roi, lines


def region_of_interest(frame, vertices):
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
    cv2.fillPoly(mask, vertices, 255)
    masked_frame = cv2.bitwise_and(frame, mask)
    return masked_frame


