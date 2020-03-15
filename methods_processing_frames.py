import cv2
import numpy as np
from self_driving_car_with_python.methods_drawing_lines import draw_closest_line, make_steering_angle, \
    draw_heading_line, make_steering_angle_default
import logging

screen_height = 800
screen_height = 640


def pre_process_frame(frame):
    """
    :param frame then convert it Gray, from there
    blur the given image and apply Canny effect
    :return: Return it with made changes
    """
    # global l1, l2, t1, t2
    steering_angle = 90
    original_frame = frame
    vertices = np.array([[10, 440], [10, 320], [200, 280], [600, 280], [800, 320], [800, 440]], np.int32)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # gray image
    canny_effect_frame = cv2.Canny(gray_frame, 100, 200)  # applied canny effect
    blur_frame = cv2.GaussianBlur(canny_effect_frame, (5, 5), 0)
    frame_with_roi = region_of_interest(blur_frame, [vertices])
    lines = cv2.HoughLinesP(frame_with_roi,
                            rho=1,
                            theta=np.pi / 180,
                            threshold=180,
                            lines=np.array([]),
                            minLineLength=30,
                            maxLineGap=15)
    try:
        l1, l2 = draw_closest_line(lines)  # , left, right
        steering_angle = make_steering_angle(frame, l1, l2)
        x1, y1, x2, y2 = draw_heading_line(frame, steering_angle)

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
                    # steering_angle = make_steering_angle_default(frame, lines)
                except Exception as e:
                    logging.error("Couldn't draw lines with given input")
                    print(str(e))
    except Exception as e:
        logging.error("Didn't get lines or line")
        print(str(e))
    # TODO kako ne bi puko ako l1 i l2 budi prazni pa daj default lajnu kako bi usporedivali za steering logic kasnije
    # if len(l1) == 0 and len(l2) == 0:
    #     need_this_1 = t1
    #     need_this_2 = t2
    # else:
    # need_this_1 = l1
    # need_this_2 = l2

    return frame_with_roi, original_frame, steering_angle  # , need_this_1, need_this_2


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
