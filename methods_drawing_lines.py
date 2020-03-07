import cv2
import numpy as np


def display_lines(frame, lines):
    """
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
    """
    #try:
    line_frame = np.zeros_like(frame)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
           # x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_frame, (x1, y1), (x2, y2), (255, 0, 0), 10)

        return line_frame
    #except Exception as e:
     #   print(traceback.format_exc())


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


