import os
from collections import Counter
from random import shuffle
from self_driving_car_with_python.methods_directions import *
import logging
from self_driving_car_with_python.alexnet import alexnet
from self_driving_car_with_python.get_keys import key_check, keys_to_output_complex
from self_driving_car_with_python.methods_processing_frames import make_gray_frame_custDimensions, pre_process_frame
from self_driving_car_with_python.methods_capturing_frames import take_frame
import time
import pandas as pd
from tensorflow import keras
import cv2
import tensorflow as tf

SCREEN_WIDTH_160 = 160
SCREEN_HEIGHT_120 = 120
GEORGE = "george_xception_5-0.1"
COSMO = "cosmo_xception_10-0.2"

LR = 1e-3
EPOCHS = 5
EPOCH_COUNT = 5
MODEL_NAME = "pygta5-car-{}-{}-{}-epochs.model".format(LR, "alexnetv3", EPOCHS)
LAST_MODEL = "pygta5-car-{}-{}-{}-epochs.model".format(LR, "alexnetv3", EPOCH_COUNT)
NEW_MODEL = "cosmo-not_shuffled-{}-{}-{}-epochs.model".format(LR, "alexnetv7", EPOCHS)
LOAD_MODEL = False


def check_existing_training_data():
    starting_value = 126
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


def make_training_data(file_name):
    file_name = file_name
    starting_value = 126
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
                time.sleep(1)
            else:
                print("Pausing")
                pause = True
                time.sleep(1)


def create_xception_model():
    new_model = tf.keras.applications.Xception(include_top=False,
                                               input_shape=(120, 160, 3))
    #  include_top - excluded the last layer of Xception model classification
    # for layer in new_model.layers:
    #     layer.trainable = False

    # Adding new layer at the end with 9 classification
    x = new_model.output
    x1 = tf.keras.layers.GlobalAveragePooling2D()(x)

    x2 = tf.keras.layers.Dense(1024, activation="relu")(x1)
    predictions = tf.keras.layers.Dense(9, activation="softmax")(x2)

    model_xception = keras.models.Model(inputs=new_model.input, outputs=predictions)

    learning_rate = 0.001
    opt = keras.optimizers.Adam(lr=learning_rate, decay=1e-3)

    model_xception.compile(loss="categorical_crossentropy",
                           optimizer=opt,
                           metrics=["accuracy"])
    # FOR MODEL ARCHITECTURE
    # print(model_xception.summary())

    return model_xception


def train_xception_model():
    # XCEPTION MODEL
    xception_model = create_xception_model()

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs/{}'.format("george_xception_5-0.1"))

    if LOAD_MODEL:
        xception_model.load_weights("george_xception_5-0.1")
        print("model loaded: {}".format("george_xception_5-0.1"))

    for epoch in range(EPOCHS):
        data_order = [i for i in range(1, 215)]
        shuffle(data_order)
        for j in data_order:
            training_data = np.load('training_data160120-{}.npy'.format(j), allow_pickle=True)
            print('Next for fit: training_data-{}.npy'.format(j),
                  "Samples: {}".format(len(training_data)),
                  "Epoch: {}".format(epoch))
            shuffle(training_data)

            rgb_train_x = []
            for x in training_data:
                rgb_image = cv2.cvtColor(x[0], cv2.COLOR_GRAY2BGR)
                rgb_train_x.append(rgb_image)
            rgb_train_x = np.array([i for i in rgb_train_x]).reshape(-1, 120, 160, 3) / 255.0

            y_train = [i[1] for i in training_data]

            xception_model.fit(np.array(rgb_train_x), np.array(y_train), batch_size=12, epochs=1, validation_split=0.1,
                               callbacks=[tensorboard])
            xception_model.save(filepath="george_xception_5-0.1")


def run_exception_model():
    model = keras.models.load_model(GEORGE)
    paused = False
    for i in list(range(4))[::-1]:
        print(i + 1)
        time.sleep(1)

    # TODO tweak the prediction
    while True:
        if not paused:
            frame = take_frame()
            frame = make_gray_frame_custDimensions(frame, 160, 120)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            # _, _, _, l1, l2 = pre_process_frame(frame) # TODO stay in lane whatever happens
            frame = frame.reshape(1, 120, 160, 3) / 255.0
            prediction = model.predict([frame])[0]
            # print(prediction)
                                                   #     w    s    a    d    wa   wd   sa  sd   nk
            prediction = np.array(prediction) * np.array([1, 1.3, 1, 1, 1, 1, 1, 1, 1])
            # print(prediction)
            make_direction_logic(prediction)

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


# OLD FUNCTIONS #


def train_model():
    model = alexnet(width=120, height=160, lr=LR, output=9)

    if LOAD_MODEL:
        model.load(LAST_MODEL)
        print("Last model loaded: {}".format(LAST_MODEL))

    for i in range(EPOCHS):
        data_order = [i for i in range(5, 60)]
        shuffle(data_order)
        for j in data_order:
            training_data = np.load('training_data160120-{}.npy'.format(j), allow_pickle=True)
            print('training_data-{}.npy'.format(j), len(training_data))

            shuffle(training_data)

            train = training_data[:-50]
            test = training_data[-50:]

            X = np.array([i[0] for i in train]).reshape((-1, 120, 160, 1))
            Y = [i[1] for i in train]

            test_x = np.array([i[0] for i in test]).reshape((-1, 120, 160, 1))
            test_y = [i[1] for i in test]

            model.fit({"input": X}, {"targets": Y}, n_epoch=1,
                      validation_set=({"input": test_x}, {"targets": test_y}),
                      snapshot_step=2500, show_metric=True, run_id=NEW_MODEL)
            model.save(NEW_MODEL)
            print("Model saved: {}".format(NEW_MODEL))


def make_training_data_old(training_data, file_name):
    while True:
        frame = take_frame()
        resized_frame = make_gray_frame_custDimensions(frame, 160, 120)
        keys = key_check()
        output = keys_to_output_complex(keys)  # keys_to_output_AWD(keys)
        training_data.append([resized_frame, output])

        if len(training_data) % 500 == 0:
            print(len(training_data))
            np.save(file_name, training_data, allow_pickle=True)


def check_for_existing_training_data_old(file_name):
    if os.path.isfile(file_name):
        print("file exist, loading previous data")
        training_data = list(np.load(file_name, allow_pickle=True))
    else:
        print("file doesnt exist, starting fresh")
        training_data = []
    return training_data


def shuffle_training_data_old(training_data):
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


def train_data_old(shuffled_training_data, screen_width, screen_height):
    model = alexnet(screen_width, screen_height, LR)
    train = shuffled_training_data[:-50]
    test = shuffled_training_data[-50:]

    X = np.array([i[0] for i in train]).reshape(-1, 160, 120, 1)
    Y = [i[1] for i in train]

    test_x = np.array([i[0] for i in test]).reshape(-1, 160, 120, 1)
    test_y = [i[1] for i in test]

    model.fit({"input": X}, {"targets": Y}, n_epoch=EPOCHS, validation_set=({"input": test_x}, {"targets": test_y}),
              snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

    # tensorboard --logdir=foo:C:/path/to/log

    model.save(MODEL_NAME)
