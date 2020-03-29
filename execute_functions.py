from self_driving_car_with_python.methods_capturing_frames import take_frame
from self_driving_car_with_python.methods_directions import steering_logic, go_right, go_left, go_straight, go_lil_left, \
    go_lil_right
from self_driving_car_with_python.methods_processing_frames import pre_process_frame, modify_frame_160x120gray, modify_frame_480x270gray
from self_driving_car_with_python.methods_neural_network import *
from self_driving_car_with_python.get_keys import key_check, keys_to_output_complex
from self_driving_car_with_python.direct_keys import PressKey, ReleaseKey, W, A, S, D
import time
import numpy as np
import warnings
import cv2

warnings.simplefilter('ignore', np.RankWarning)
np.seterr(divide='ignore', invalid='ignore')


def execute_create_training_data():
    #file_name = 'training_data.npy'  # new training data
    # training_data = check_for_existing_training_data(file_name=file_name)
    # make_training_data(file_name=file_name, training_data=training_data)
    file_name = check_existing_training_data_new()
    make_training_data_new(file_name=file_name)


def execute_shuffle_data():
    training_data = np.load('training_data.npy', allow_pickle=True)  # shuffle new training data
    shuffle_training_data(training_data)  # shuffle new training data


def execute_to_train_data():
    shuffled_training_data = np.load('training_data_shuffled.npy', allow_pickle=True)  # shuffle new training data
    train_data(shuffled_training_data)  # train new training data


def execute_model():
    paused = True
    model = run_model()
    time_it_3()
    if not paused:
        frame = take_frame()
        resized_screen = modify_frame_160x120gray(frame)

        prediction = model.predict([resized_screen.reshape(SCREEN_WIDTH_160, SCREEN_HEIGHT_120, 1)])[0]
        print(prediction)

        turn_thresh = .75
        fwd_thresh = 0.70

        if prediction[1] > fwd_thresh:
            go_lil_left()
        elif prediction[0] > turn_thresh:
            go_straight()
        elif prediction[2] > turn_thresh:
            go_lil_right()
        # else:
        #     go_straight()

    keys = key_check()

    if 'T' in keys:
        if paused:
            paused = False
            time.sleep(1)
        else:
            paused = True
            ReleaseKey(A)
            ReleaseKey(W)
            ReleaseKey(D)
            time.sleep(1)


def execute_self_driving_cv():
    time_it_3()
    while True:
        frame = take_frame()
        processed_frame, original_frame, steering_angle, l1, l2 = pre_process_frame(frame=frame)  # left, right
        steering_logic(steering_angle, l1, l2)
        cv2.imshow("window2", original_frame)
        cv2.imshow("processed", processed_frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


def time_it_3():
    for i in list(range(3))[::-1]:
        print(i + 1)
        time.sleep(1)
