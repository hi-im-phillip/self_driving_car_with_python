import cv2
import numpy as np
from self_driving_car_with_python.methods_drawing_lines import draw_closest_line, draw_line

screen_height = 800
screen_height = 640


def pre_process_frame(frame):
    """
    :param frame then convert it Gray, from there
    blur the given image and apply Canny effect
    :return: Return it with made changes
    """
    original_frame = frame
    vertices = np.array([[10, 440], [10, 300], [200, 260], [600, 260], [800, 300], [800, 440]], np.int32)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # gray image
    canny_effect_frame = cv2.Canny(gray_frame, 100, 200)  # applied canny effect
    blur_frame = cv2.GaussianBlur(canny_effect_frame, (5, 5), 0)
    frame_with_roi = region_of_interest(blur_frame, [vertices])
    lines = cv2.HoughLinesP(frame_with_roi,
                            rho=1,
                            theta=np.pi/180,
                            threshold=150,
                            lines=np.array([]),
                            minLineLength=20,
                            maxLineGap=15)
    try:
        l1, l2 = draw_closest_line(original_frame, lines) # , left, right
        cv2.line(original_frame, (l1[0], l1[1]), (l1[2], l1[3]), [0, 250, 0], 30)
        cv2.line(original_frame, (l2[0], l2[1]), (l2[2], l2[3]), [0, 250, 0], 30)
    except Exception as e:
        print(str(e))
        pass
    try:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2 == x1:
                    continue
                try:
                    cv2.line(frame_with_roi, (x1, y1), (x2, y2), [250, 0, 0], 3)
                except Exception as e:
                    print(str(e))
    except Exception as e:
        print(str(e))

    return frame_with_roi, lines, original_frame#, left, right


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


