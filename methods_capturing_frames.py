import cv2
import numpy as np
import time
import warnings
from self_driving_car_with_python.grab_screen import grab_screen
from self_driving_car_with_python.direct_keys import PressKey, ReleaseKey, W, A, S, D
import traceback

screen_height = 800
screen_width = 640

warnings.simplefilter('ignore', np.RankWarning)
np.seterr(divide='ignore', invalid='ignore')


def make_coordinates(frame, line_parameters):
    slope, intercept = line_parameters
    y1 = frame.shape[0]
    y2 = int(330) #int(y1*(2/3))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(frame, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    try:
        left_line = make_coordinates(frame, left_fit_average)
        right_line = make_coordinates(frame, right_fit_average)
        return np.array([left_line, right_line]), right_line, left_line
    except Exception as e:
        print(e, '\n')
        return None


def region_of_interest(frame):
    '''
    :param Takes a image and return enclosed region of field of view,
    and enclosed region is triangle shape
    triangle shape - 200, bottom - 700, bottom - 400, 200
    mask - create same shape of black mask as give frame
    fillPolly - take a triangle(polygons) which borders are defined and apply it on mask and that area will be white
    :return: masked_frame - computing the bitwise & of both images, ultimately masking the canny image to only show the
    region of interest traced by the polygonal contour of the mask
    '''
    mask = np.zeros_like(frame)
    vertices = np.array([[[0, 600], [0, 400], [200, 320], [600, 320], [screen_height, 400], [screen_height, 600]]]) #np.array([[[50, 475], [0, 600], [0, 350], [200, 300], [600, 300], [800, 350], [800, 600], [750, 475]]]) # np.array([[(100, height), (700, height), (400, 100)]])
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
    lines = cv2.HoughLinesP(cropped_canny_frame, 1, np.pi/180, 180, 20, 15)  #cv2.HoughLinesP(cropped_canny_frame, 2, np.pi/180, 180, np.array([]), minLineLength=40, maxLineGap=5)
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
    #try:
    line_frame = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_frame, (x1, y1), (x2, y2), (255, 0, 0), 10)

        return line_frame
    #except Exception as e:
     #   print(traceback.format_exc())


def pre_process_frame(frame):
    '''
    :param frame then convert it Gray, from there
    blur the given image and apply Canny effect
    :return: Return it with made changes
    '''
    processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    processed_frame = cv2.Canny(processed_frame, 200, 350)
    processed_frame = cv2.GaussianBlur(processed_frame, (5, 5), 0)
    processed_frame = region_of_interest(frame=processed_frame)
    return processed_frame


def make_frames():
    '''
    Take a frame of the screen and store it in memory
    region - Specifies specific region (bbox = x cord, y cord, width, height)
    frame_with_lines - 2.arg: pixel intensity (make it darker), 4.arg: lines will be clearly define, 5.arg gamma arg
    '''
    while True:
        #  time_test = time.time()
        frame = grab_screen(region=(0, 0, screen_height, screen_width))
        # clean_frame = np.copy(grabbed_frame)
        processed_frame = pre_process_frame(frame=frame)
        lines = detect_line(processed_frame)
        # average_lines = average_slope_intercept(frame, lines)
        line_frame = display_lines(frame, lines)
        frame_with_lines = cv2.addWeighted(frame, 0.8, line_frame, 1, 1)
        cv2.imshow("window", frame_with_lines)

        # TODO
        # processed frame -> 0 0 0 0
        # line.reshape take 1 instead of 4
        # cv2.addWeight arg is neither array op array

        # cv2.imshow("window", cropped_canny_frame)
        # print(time.time() - time_test)
        # if average_lines[0] > 0 and average_lines[1] > 0:
        #     turn_left()
        # elif average_lines[0] < 0 and average_lines[1] < 0:
        #     turn_right()
        # else:
        #     straight()
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            cv2.destroyAllWindows()
            break

