from self_driving_car_with_python.methods_capturing_frames import take_frame
from self_driving_car_with_python.methods_directions import steering_logic
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
    #  TODO if both of line are on the same side, turn opposite direction
    #  TODO reduce detection of lines 
    while True:
        # time_test = time.time()
        frame = take_frame()
        processed_frame, original_frame, steering_angle, canny_effect_frame, l1, l2 = pre_process_frame(frame=frame)  # left, right
        steering_logic(steering_angle, l1, l2)
        # print(steering_angle)
        cv2.imshow("window2", original_frame)
        cv2.imshow("processed", processed_frame)
        #cv2.imshow("window", canny_effect_frame)
        # print(time.time() - time_test)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break





