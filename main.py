from self_driving_car_with_python.methods_capturing_frames import take_frame
from self_driving_car_with_python.direct_keys import PressKey, ReleaseKey, W, A, S, D
from self_driving_car_with_python.methods_drawing_lines import average_slope_intercept, display_lines
from self_driving_car_with_python.methods_processing_frames import pre_process_frame
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
        processed_frame = pre_process_frame(frame=frame)
        lines = cv2.HoughLinesP(processed_frame,
                                rho=2,
                                theta=np.pi / 180,
                                threshold=180,
                                lines=np.array([]),
                                maxLineGap=15)
        average_lines = average_slope_intercept(frame=processed_frame, lines=lines)
        line_frame = display_lines(frame=processed_frame, lines=average_lines)

        frame_with_lines = cv2.addWeighted(processed_frame, 0.8, line_frame, 1, 1)
        cv2.imshow("window", frame_with_lines)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            cv2.destroyAllWindows()
            break
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




