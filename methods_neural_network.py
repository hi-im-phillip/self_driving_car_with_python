import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
from self_driving_car_with_python.methods_directions import *
import logging
from self_driving_car_with_python.alexnet import alexnet, alexnet2
from self_driving_car_with_python.get_keys import keys_to_output_AWD, key_check, keys_to_output_complex, w, a, s, d, wd, wa, sd, sa, nk
from self_driving_car_with_python.methods_processing_frames import make_gray_frame_custDimensions, \
    modify_frame_480x270gray
from self_driving_car_with_python.methods_capturing_frames import take_frame
import time
import pandas as pd

SCREEN_WIDTH_160 = 160
SCREEN_HEIGHT_120 = 120
SCREEN_WIDTH_480 = 480
SCREEN_HEIGHT_270 = 270
LR = 1e-3
EPOCH = 10
MODEL_NAME = "pygta5-car-{}-{}-{}-epochs.model".format(LR, "alexnetv3", EPOCH)


def check_existing_training_data_new():
    starting_value = 28
    file_name = "training_data160120-{}.npy".format(starting_value)
    while True:
        if os.path.isfile(file_name):
            print("File exist, iterating next")
            starting_value += 1
            file_name = "training_data160120-{}.npy".format(starting_value)
        else:
            print("file doesnt exist, starting new", starting_value)
            file_name = "training_data160120-{}.npy".format(starting_value)
            return file_name


def make_training_data_new(file_name):
    file_name = file_name
    starting_value = 28
    training_data = []
    pause = False
    while True:
        if not pause:
            frame = take_frame()
            resized_frame = make_gray_frame_custDimensions(frame, 160, 120)
            keys = key_check()
            output = keys_to_output_complex(keys)
            training_data.append([resized_frame, output])

            if len(training_data) % 100 == 0:
                print(len(training_data))

                if len(training_data) == 500:
                    np.save(file_name, training_data, allow_pickle=True)
                    print("Training data saved")
                    training_data = []
                    starting_value += 1
                    file_name = "training_data160120-{}.npy".format(starting_value)
                    print(file_name)
        keys = key_check()
        if "T" in keys:
            if pause:
                pause = False
                print("Resuming")
            else:
                print("Pausing")
                pause = True
                time.sleep(1)


def make_training_data(training_data, file_name):
    while True:
        frame = take_frame()
        resized_frame = make_gray_frame_custDimensions(frame, 160, 120)
        keys = key_check()
        output = keys_to_output_complex(keys)  # keys_to_output_AWD(keys)
        training_data.append([resized_frame, output])

        if len(training_data) % 500 == 0:
            print(len(training_data))
            np.save(file_name, training_data, allow_pickle=True)


def check_for_existing_training_data(file_name):
    if os.path.isfile(file_name):
        print("file exist, loading previous data")
        training_data = list(np.load(file_name, allow_pickle=True))
    else:
        print("file doesnt exist, starting fresh")
        training_data = []
    return training_data


def shuffle_training_data(training_data):
    print(len(training_data))
    df = pd.DataFrame(training_data)
    print(df.head())
    print(Counter(df[1].apply(str)))

    left_turns = []
    right_turns = []
    straights = []

    shuffle(training_data)

    for data in training_data:
        frame = data[0]
        turns = data[1]

        if turns == [1, 0, 0]:
            left_turns.append([frame, turns])
        elif turns == [0, 1, 0]:
            straights.append([frame, turns])
        elif turns == [0, 0, 1]:
            right_turns.append([frame, turns])
        else:
            logging.warning("Turn didn't match")

    straights = straights[:len(left_turns)][:len(right_turns)]
    left_turns = left_turns[:len(straights)]
    right_turns = right_turns[:len(right_turns)]

    final_data = straights + left_turns + right_turns
    shuffle(final_data)

    print(len(final_data))
    np.save("training_data_shuffled.npy", final_data)


def train_data(shuffled_training_data, screen_width, screen_height):
    model = alexnet(screen_width, screen_height, LR)
    train = shuffled_training_data[:-50]
    test = shuffled_training_data[-50:]

    X = np.array([i[0] for i in train]).reshape(-1, 160, 120, 1)
    Y = [i[1] for i in train]

    test_x = np.array([i[0] for i in test]).reshape(-1, 160, 120, 1)
    test_y = [i[1] for i in test]

    model.fit({"input": X}, {"targets": Y}, n_epoch=EPOCH, validation_set=({"input": test_x}, {"targets": test_y}),
              snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

    # tensorboard --logdir=foo:C:/path/to/log

    model.save(MODEL_NAME)


def train_data_new():
    model = alexnet2(width=SCREEN_WIDTH_160, height=SCREEN_HEIGHT_120, lr=LR, output=9)
    for i in range(EPOCH):
        data_order = [i for i in range(1, 53)]
        shuffle(data_order)
        for j in data_order:
            training_data = np.load('training_data160120-{}.npy'.format(j), allow_pickle=True)
            print('training_data-{}.npy'.format(j), len(training_data))

            df = pd.DataFrame(training_data)
            df = df.iloc[np.random.permutation(len(df))]
            training_data = df.values.tolist()

            train = training_data[:-50]
            test = training_data[-50:]

            X = np.array([i[0] for i in train]).reshape((-1, 160, 120, 1))
            Y = [i[1] for i in train]

            test_x = np.array([i[0] for i in test]).reshape((-1, 160, 120, 1))
            test_y = [i[1] for i in test]

            model.fit({"input": X}, {"targets": Y}, n_epoch=1,
                      validation_set=({"input": test_x}, {"targets": test_y}),
                      snapshot_step=2500, show_metric=True, run_id=MODEL_NAME)
            model.save(MODEL_NAME)
            print("Model saved")


def test_model():
    model = alexnet2(width=SCREEN_WIDTH_160, height=SCREEN_HEIGHT_120, lr=LR, output=9)
    model.load(MODEL_NAME)
    paused = False
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    # w = [1, 0, 0, 0, 0, 0, 0, 0, 0] 1
    # s = [0, 1, 0, 0, 0, 0, 0, 0, 0] 2
    # a = [0, 0, 1, 0, 0, 0, 0, 0, 0] 3
    # d = [0, 0, 0, 1, 0, 0, 0, 0, 0] 4
    # wa = [0, 0, 0, 0, 1, 0, 0, 0, 0] 5
    # wd = [0, 0, 0, 0, 0, 1, 0, 0, 0] 6
    # sa = [0, 0, 0, 0, 0, 0, 1, 0, 0] 7
    # sd = [0, 0, 0, 0, 0, 0, 0, 1, 0] 8
    # nk = [0, 0, 0, 0, 0, 0, 0, 0, 1] 9
    # TODO tweak the prediction
    while True:
        if not paused:
            frame = take_frame()
            frame = make_gray_frame_custDimensions(frame, 160, 120)
            prediction = model.predict([frame.reshape(160, 120, 1)])[0]
                                                      #    w    s    a    d    wa   wd   sa  sd   nk
            prediction = np.array(prediction) * np.array([0.4, 1.3, 1.3, 1.3, 1.1, 1.1, 0.5, 0.5, 0.2])
            print(prediction)
            if np.argmax(prediction) == np.argmax(w):
            # if prediction[0] > 0.60:
                go_straight()
                print("straight")
            elif np.argmax(prediction) == np.argmax(s):
            # elif prediction[1] >
                reverse()
                print("reverse")
            elif np.argmax(prediction) == np.argmax(a):
                go_left()
                print("left")
            elif np.argmax(prediction) == np.argmax(d):
                go_right()
                print("right")
            elif np.argmax(prediction) == np.argmax(wa):
                forward_left()
                print("forward_left")
            elif np.argmax(prediction) == np.argmax(wd):
                forward_right()
                print("forward_right")
            elif np.argmax(prediction) == np.argmax(sa):
                reverse_left()
                print("reverse_left")
            elif np.argmax(prediction) == np.argmax(sd):
                reverse_right()
                print("reverse_right")
            elif np.argmax(prediction) == np.argmax(nk):
                no_keys()
                print("no_keys")

        keys = key_check()
        if "T" in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                time.sleep(1)


def run_model():
    model = alexnet(SCREEN_WIDTH_160, SCREEN_HEIGHT_120, LR)
    model.load(MODEL_NAME)
    return model
