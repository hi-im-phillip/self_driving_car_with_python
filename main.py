from self_driving_car_with_python.methods_capturing_frames import take_frame
from self_driving_car_with_python.methods_directions import go_left, go_right, go_straight
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
        #  time_test = time.time()
        frame = take_frame()
        processed_frame, lines, original_frame = pre_process_frame(frame=frame) # , left, right
        # if left > (-700) and right > 1900:
        #     go_right()
        # elif left < (-1300) and right < 800:
        #     go_left()
        # else:
        #     go_straight()
        cv2.imshow("window", original_frame)
        cv2.imshow("processed", processed_frame)
        #cv2.imshow("window", draw_closest_line(frame, lines))
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break





