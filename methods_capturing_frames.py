import cv2
import numpy as np
import time
from self_driving_car_with_python.grab_screen import grab_screen


def region_of_interest(frame):
    '''
    :param Takes a image and return enclosed region of field of view,
    and enclosed region is triangle shape
    height - height of the given frame, x-cord[0]
    triangle shape - 200, bottom - 700, bottom - 400, 200
    mask - create same shape of black mask as give frame
    fillPolly - take a triangle(polygons) which borders are defined and apply it on mask and that area will be white
    :return: masked_frame - computing the bitwise & of both images, ultimately masking the canny image to only show the
    region of interest traced by the polygonal contour of the mask
    '''
    height = frame.shape[0]
    mask = np.zeros_like(frame)
    vertices = np.array([[[0, 600], [0, 380], [200, 330], [600, 330], [800, 380], [800, 600]]], np.int32)
    cv2.fillPoly(mask, vertices, 255)
    masked_frame = cv2.bitwise_and(frame, mask)
    return masked_frame


def detect_line(cropped_canny_frame):
    '''
    :param cropped_canny_frame: enclosed region of field of view
    :return:
    lines args 2. 2 degree precision
    3. Defining radian precision resolution, 1 radian
    4. threshold: minimum number of votes needed to accept a candidate line
    5. Placeholder
    6. length of line in pixels (less then 40 px are rejected)
    7. Maximum distance between segmented lines which will be allowed to connect in single line (not to be broken up)
    '''
    lines = cv2.HoughLinesP(cropped_canny_frame, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    return lines


def display_lines(frame, lines):
    '''
    :param frame: frame on which will be lines displayed
    :param lines: 3d array
    np.zeros_like() - declare array of zeros of the same dimension of frame(array), pixels will be completly black
    line - each line is a 2D array containing line coordinate in the form [[x1, y1, x2, x2]]
    These coordinates specify the line's parameters, as well as the location of the lines with respect
    to the image space, ensuring that they are placed in the correct position
    line.reshape() - reshaping in 1d array from 2d (creating 4 elements and store them in variables(elements))
    :return: cv2.line args - 1. draw it on line_frame (black image)
    2.,3. which coordinates in image space draw a lines
    4. color of lines
    5. line thickness - higher the thicker
    '''
    line_frame = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_frame, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_frame


def make_canny_frame(frame):
    '''
    :param Take a frame and then convert it Gray, from there
    blur the given image and apply Canny effect
    :return: Return it with made changes
    '''
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    canny_frame = cv2.Canny(blur_frame, 50, 150)
    return canny_frame


def make_frames():
    '''
    Take a frame of the screen and store it in memory
    region - Specifies specific region (bbox = x cord, y cord, width, height)
    frame_with_lines - 2.arg: pixel intensity (make it darker), 4.arg: lines will be clearly define, 5.arg gamma arg
    '''
    while True:
        time_test = time.time()
        frame = grab_screen(region=(0, 0, 800, 640))
        canny_frame = make_canny_frame(frame=frame)
        cropped_canny_frame = region_of_interest(frame=canny_frame)
        line_frame = display_lines(frame, detect_line(cropped_canny_frame))
        frame_with_lines = cv2.addWeighted(frame, 0.8, line_frame, 1, 1)
        cv2.imshow("window", frame_with_lines)
        print(time.time() - time_test)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            cv2.destroyAllWindows()
            break

