import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import cv2
import logging
from self_driving_car_with_python.alexnet import alexnet
from self_driving_car_with_python.get_keys import keys_to_output_AWD, key_check, keys_to_output_complex
from self_driving_car_with_python.methods_processing_frames import modify_frame_160x120gray, modify_frame_480x270gray
from self_driving_car_with_python.methods_capturing_frames import take_frame
import os
import time

SCREEN_WIDTH_160 = 160
SCREEN_HEIGHT_120 = 120
SCREEN_WIDTH_480 = 480
SCREEN_HEIGHT_270 = 270
LR = 1e-3
EPOCH = 10
MODEL_NAME = "pygta5-car-{}-{}-{}-epochs.model".format(LR, "alexnetv2", EPOCH)


def check_existing_training_data_new():
    starting_value = 1
    file_name = "training_data-{}.npy".format(starting_value)
    while True:
        if os.path.isfile(file_name):
            print("File exist, iterating next")
            starting_value += 1
            file_name = "training_data-{}.npy".format(starting_value)
        else:
            print("file doesnt exist, starting new")
            return file_name


def make_training_data_new(file_name):
    file_name = file_name
    starting_value = 1
    training_data = []
    pause = False
    while True:
        if not pause:
            frame = take_frame()
            resized_frame = modify_frame_480x270gray(frame)
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
                    file_name = "training_data-{}.npy".format(starting_value)
        keys = key_check()
        if "T" in keys:
            if pause:
                pause = False
                print("Resuming")
            else:
                print("Pausing")
                pause = True
                time.sleep(1)


def make_training_data(frame, training_data, file_name):
    resized_frame = modify_frame_160x120gray(frame)
    keys = key_check()
    output = keys_to_output_AWD(keys)
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


def train_data(shuffled_training_data):

    model = alexnet(SCREEN_WIDTH_160, SCREEN_HEIGHT_120, LR)
    train = shuffled_training_data[:-50]
    test = shuffled_training_data[-50:]

    X = np.array([i[0] for i in train]).reshape(-1, SCREEN_WIDTH_160, SCREEN_HEIGHT_120, 1)
    Y = [i[1] for i in train]

    test_x = np.array([i[0] for i in test]).reshape(-1, SCREEN_WIDTH_160, SCREEN_HEIGHT_120, 1)
    test_y = [i[1] for i in test]

    model.fit({"input": X}, {"targets": Y}, n_epoch=EPOCH, validation_set=({"input": test_x}, {"targets": test_y}),
              snapshot_step=2500, show_metric=True, run_id=MODEL_NAME)

    # tensorboard --logdir=foo:C:/path/to/log

    model.save(MODEL_NAME)


def train_data_new():
    model = alexnet(width=SCREEN_WIDTH_480, height=SCREEN_HEIGHT_270)
    starting_value = 1
    hm_data = 22
    for i in range(EPOCH):
        for i in range(1, hm_data+1):
            training_data = np.load('training_data.npy'.format(i+1))

            train = training_data[:-100]
            test = training_data[-100:]

            X = np.array([i[0] for i in train]).reshape(-1, SCREEN_WIDTH_480, SCREEN_HEIGHT_270, 1)
            Y = [i[1] for i in train]

            test_x = np.array([i[0] for i in test]).reshape(-1, SCREEN_WIDTH_480, SCREEN_HEIGHT_270, 1)
            test_y = [i[1] for i in test]

            model.fit({'input': X}, {'targets': Y}, n_epoch=1, validation_set=({'input': test_x}, {'targets': test_y}),
                      snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
            model.save(MODEL_NAME)


def run_model():
    model = alexnet(SCREEN_WIDTH_160, SCREEN_HEIGHT_120, LR)
    model.load(MODEL_NAME)
    return model


