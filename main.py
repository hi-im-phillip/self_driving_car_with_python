from self_driving_car_with_python.methods_capturing_frames import take_frame
from self_driving_car_with_python.methods_directions import go_left, go_right, go_straight, go_lil_left, go_lil_right
from self_driving_car_with_python.methods_processing_frames import pre_process_frame
from self_driving_car_with_python.methods_drawing_lines import draw_closest_line, draw_line
import time
import numpy as np
import warnings
import cv2


warnings.simplefilter('ignore', np.RankWarning)
np.seterr(divide='ignore', invalid='ignore')


if __name__ == '__main__':

    for i in list(range(3))[::-1]:
        print(i + 1)
        time.sleep(1)

    while True:
        # time_test = time.time()
        frame = take_frame()
        processed_frame, original_frame, steering_angle = pre_process_frame(frame=frame)  # left, right

        if 40 > steering_angle and steering_angle < 80:
            go_lil_left()
            print("going lil_left")
            print(steering_angle)
        elif 100 > steering_angle and steering_angle < 140:
            go_lil_right()
            print("going_lil_right")
            print(steering_angle)
        elif steering_angle > 140:
            go_right()
            print("going_right")
            print(steering_angle)
        elif steering_angle < 40:
            go_left()
            print("going_left")
            print(steering_angle)
        else:
            go_straight()
            print("going straight")
            print(steering_angle)
        cv2.imshow("window", original_frame)
        cv2.imshow("processed", processed_frame)
        # cv2.imshow("window", draw_closest_line(frame, lines))
        # print(time.time() - time_test)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break





