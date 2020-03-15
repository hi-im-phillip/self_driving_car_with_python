import cv2
import numpy as np
import math
import logging


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
    line_frame = np.zeros_like(frame)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_frame, (x1, y1), (x2, y2), (255, 0, 0), 10)
        return line_frame


def make_coordinates(frame, line_parameters):
    slope, intercept = line_parameters
    y1 = frame.shape[0]
    y2 = int(330)
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


def draw_line(frame, lines, color=[255, 0, 0], thickness=3):
    if lines is None:
        return
    frame = np.copy(frame)
    line_img = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8, )

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(frame, (x1, y1), (x2, y2), [255, 255, 255], 3)

    frame = cv2.addWeighted(frame, 0.8, line_img, 1.0, 0.0)
    return frame


def draw_closest_line(lines):

    try:
        left_line_x = []
        left_line_y = []
        right_line_x = []
        right_line_y = []

        for line in lines:
            for x1, y1, x2, y2 in line:
                if x1 == x2:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if math.fabs(slope) < 0.05:
                    continue
                if slope < 0:  # negative slope is left line
                    left_line_x.extend([x1, x2])
                    left_line_y.extend([y1, y2])
                else:
                    right_line_x.extend([x1, x2])
                    right_line_y.extend([y1, y2])

        # min_y_left = 300 #int(frame.shape[0] * (3/5))
        # max_y_left = int(frame.shape[0])

        # Definition of the new line length
        ys = []
        for line in lines:
            for xy in line:
                ys += [xy[1], xy[3]]
        max_y_left = 600
        min_y_left = min(ys)

        max_y_right = max_y_left
        min_y_right = min_y_left
        #  if one of lines is empty, set that lane to 0
        if len(left_line_x) == 0 and len(left_line_y) == 0:
            left_x_start = 0
            left_x_end = 0
            max_y_left = 0
            min_y_left = 0
        else:
            poly_left = np.poly1d(np.polyfit(left_line_y, left_line_x, deg=1))
            left_x_start = int(poly_left(max_y_left))
            left_x_end = int(poly_left(min_y_left))

        if len(right_line_y) == 0 and len(right_line_x) == 0:
            right_x_start = 0
            right_x_end = 0
            max_y_right = 0
            min_y_right = 0
        else:
            poly_right = np.poly1d(np.polyfit(right_line_y, right_line_x, deg=1))
            right_x_start = int(poly_right(max_y_right))
            right_x_end = int(poly_right(min_y_right))

        # TODO Logic steering
        # print("left")
        # print(left_x_end)
        # print("right")
        # print(right_x_start - right_x_end)
        # print("\n")
        # left = left_x_start - left_x_end
        # right = right_x_start - right_x_end
        # print(int(mean(right_line_x)))
        # print(int(mean(left_line_x)))
        #print(slope)
        # frame_with_closest_line = draw_line(frame, [[
        #     [left_x_start, max_y_left, left_x_end, min_y_left],
        #     [right_x_start, max_y_left, right_x_end, min_y_left],
        # ]], thickness=5, )

        return [left_x_start, max_y_left, left_x_end, min_y_left], [right_x_start, max_y_right, right_x_end, min_y_right]
        # return frame_with_closest_line
    except Exception as e:
        print(str(e))


def make_steering_angle(frame, l1, l2):
    height, width, _ = frame.shape

    if l1 == [0, 0, 0, 0]:
        x1, _, x2, _ = l2
        x_offset = x2 - x1
    elif l2 == [0, 0, 0, 0]:
        x1, _, x2, _ = l1
        x_offset = x2 - x1
    else:
        left_x2 = l1[0]
        right_x2 = l2[0]
        mid_offset_percent = 0.02
        mid = int(width / 2 * (1 + mid_offset_percent))
        x_offset = (left_x2 + right_x2) / 2 - mid

    y_offset = int(height / 2)

    angle_to_mid_radian = math.atan(x_offset / y_offset)
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)
    steering_angle = angle_to_mid_deg + 90

    return steering_angle


def make_steering_angle_default(frame, lines):
    height, width, _ = frame.shape

    if len(lines) == 0:
        logging.info("No line in lines")
        return -90

    if len(lines) == 1:
        logging.info("Only 1 line detected")
        x1, _, x2, _ = lines[0][0]
        x_offset = x2 - x1
    else:
        _, _, left_x2, _ = lines[0][0]
        _, _, right_x2, _ = lines[1][0]
        mid_offset_percent = 0.02
        mid = int(width / 2 * (1 + mid_offset_percent))
        x_offset = (left_x2 + right_x2) / 2 - mid

    y_offset = int(height / 2)

    angle_to_mid_radian = math.atan(x_offset / y_offset)
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)
    steering_angle = angle_to_mid_deg + 90

    return steering_angle


def draw_heading_line(frame, steering_angle, color=[0, 0, 255], line_width=5):
    height, width, _ = frame.shape
    steering_angle_radian = steering_angle / 180 * math.pi
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)

    return x1, y1, x2, y2


def stabilize_steering_angle(current_angle, new_angle, max_angle_deviation=5):
    angle_deviation = new_angle - current_angle
    if abs(angle_deviation) > max_angle_deviation:
        stabilized_angle = int(current_angle + max_angle_deviation * angle_deviation / abs(angle_deviation))
    else:
        stabilized_angle = new_angle

    return stabilized_angle


def draw_line_old(img, lines, color=[255, 0, 0], thickness=3):
    if lines is None:
        return
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), [255, 255, 255], 3)