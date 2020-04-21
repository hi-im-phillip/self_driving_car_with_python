from self_driving_car_with_python.methods_capturing_frames import take_frame
from self_driving_car_with_python.methods_directions import steering_logic
from self_driving_car_with_python.methods_processing_frames import pre_process_frame
from self_driving_car_with_python.methods_neural_network import run_exception_model, train_xception_model, check_existing_training_data, make_training_data
from self_driving_car_with_python.methods_helper import timer
import numpy as np
import warnings
import cv2


warnings.simplefilter('ignore', np.RankWarning)
np.seterr(divide='ignore', invalid='ignore')


def execute_create_training_data():
    file_name = check_existing_training_data()
    make_training_data(file_name=file_name)


def execute_to_train_data():
    train_xception_model()


def execute_run_model():
    run_exception_model()


def execute_self_driving_cv():
    timer()
    while True:
        frame = take_frame()
        processed_frame, original_frame, steering_angle, l1, l2 = pre_process_frame(frame=frame)  # left, right
        steering_logic(steering_angle, l1, l2)
        cv2.imshow("window2", original_frame)
        cv2.imshow("processed", processed_frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break



