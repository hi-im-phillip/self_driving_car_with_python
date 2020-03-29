import cv2
import numpy as np
from self_driving_car_with_python.methods_drawing_lines import draw_closest_line, make_steering_angle, \
    draw_heading_line
import logging
from statistics import mean

screen_height = 800
screen_height = 640


def pre_process_frame(frame):
    """
    :param frame then convert it Gray, from there
    blur the given image and apply Canny effect
    :return: Return it with made changes
    """
    vertices_800x600 = np.array([[370, 360], [200, 465], [5, 465], [100, 350], [350, 260], [450, 260], [700, 350], [800, 465], [600, 465], [430, 360]], np.int32)
    vertices_1024x768 = np.array([[450, 460], [200, 565], [5, 565], [100, 450], [450, 260], [550, 260], [924, 450], [1024, 565], [800, 565], [550, 460]], np.int32)
    steering_angle = 90
    original_frame = frame
    vertices = vertices_1024x768  # last two sets of vertices added because of sings on road need to be cropped
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # gray image
    canny_effect_frame = cv2.Canny(gray_frame, 100, 200)  # applied canny effect
    blur_frame = cv2.GaussianBlur(canny_effect_frame, (5, 5), 0)
    frame_with_roi = region_of_interest(blur_frame, [vertices])
    lines = cv2.HoughLinesP(frame_with_roi,
                            rho=2,
                            theta=np.pi / 180,
                            threshold=130,
                            lines=np.array([]),
                            minLineLength=20,
                            maxLineGap=10)
    try:
        l1, l2 = draw_closest_line(lines)  # , left, right
        steering_angle = make_steering_angle(frame, l1, l2)
        x1, y1, x2, y2 = draw_heading_line(frame, steering_angle)
        #print(mean(l1[0::1]), mean(l2[0::1]))

        cv2.line(original_frame, (l1[0], l1[1]), (l1[2], l1[3]), [0, 250, 0], 30)
        cv2.line(original_frame, (l2[0], l2[1]), (l2[2], l2[3]), [0, 250, 0], 30)
        cv2.line(original_frame, (x1, y1), (x2, y2), [0, 0, 255], 5)

    except Exception as e:
        logging.error("Couldn't draw closest lines or heading line with given input")
        print(str(e))
        pass
    try:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2 == x1:
                    continue
                try:
                    # t1, t2 = draw_closest_line(lines)
                    cv2.line(frame_with_roi, (x1, y1), (x2, y2), [250, 0, 0], 3)
                    # print(mean(l1[0::1]), mean(l2[0::1]))
                    # steering_angle = make_steering_angle_default(frame, lines)
                except Exception as e:
                    logging.error("Couldn't draw lines with given input")
                    print(str(e))
    except Exception as e:
        logging.error("Didn't get lines or line")
        print(str(e))

    return frame_with_roi, original_frame, steering_angle, l1, l2


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


def modify_frame_160x120gray(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, (160, 120))
    return resized_frame


def modify_frame_480x270gray(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, (480, 270))
    return resized_frame

