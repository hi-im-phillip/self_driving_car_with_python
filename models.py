import tensorflow as tf
from tensorflow import keras
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

EPOCHS = 5
VERSION = 0.1
NAME = "george_xception_{}-{}".format(EPOCHS, VERSION)
NEW_NAME = "cosmo_xception_{}-{}".format(EPOCHS, VERSION)
MODEL_LOAD = True

dense_layers = [1]  # how many dense layers we want
layer_sizes = [64]
conv_layers = [2]


def create_model():
    for dense_layer in dense_layers:
        for layer_size in layer_sizes:
            for conv_layer in conv_layers:
                MODEL_NAME = "cosmo_xception_{}-conv-{}-nodes-{}-dense-{}".format(conv_layer,
                                                                                  layer_size,
                                                                                  dense_layer,
                                                                                  int(time.time()))
                my_model = tf.keras.Sequential()

                my_model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(120, 160, 1)))
                my_model.add(tf.keras.layers.Activation("relu"))
                my_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

                for layer in range(conv_layer - 1):  # - 1 because we already got one hidden layer
                    my_model.add(tf.keras.layers.Conv2D(layer_size, (3, 3)))
                    my_model.add(tf.keras.layers.Activation("relu"))
                    my_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

                my_model.add(keras.layers.Flatten())

                for layer in range(dense_layer):
                    my_model.add(tf.keras.layers.Dense(1024))
                    my_model.add(tf.keras.layers.Activation("relu"))
                    my_model.add(tf.keras.layers.Dropout(0.5))

                my_model.add(tf.keras.layers.Dense(9))
                my_model.add(tf.keras.layers.Activation('softmax'))

                my_model.compile(optimizer='adam',
                                 loss='categorical_crossentropy',
                                 metrics=['accuracy'])

    return my_model, MODEL_NAME


def test_model_w_plotting():
    test_model = tf.keras.models.load_model("sel_driving_car_model_3-conv-128-nodes-1-dense-1587122082")
    test_data = np.load("training_data160120-30.npy", allow_pickle=True)
    image_array = np.array([i[0] for i in test_data]).reshape(-1, 120, 160, 1) / 255.0
    t = 1
    for image in image_array[:200]:
        image = image.reshape(1, 120, 160, 1)
        plt.figure()
        plt.imshow(image.squeeze())
        plt.xlabel("Prediction for the {}.frame is {}".format(t, np.argmax(test_model.predict([image]))))
        # print("Prediction for the {}.frame is {}".format(t, np.argmax(model.predict([image]))))
        plt.show()
        t = t + 1


def xception_model():
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


def test_xception_model():
    test_model = keras.models.load_model("cosmo_xception_10-0.2")
    test_data = np.load("training_data160120-3.npy", allow_pickle=True)

    h = 1
    rgb_train_x_test = []
    for x in test_data:
        rgb_image_test = cv2.cvtColor(x[0], cv2.COLOR_GRAY2BGR)
        rgb_train_x_test.append(rgb_image_test)
    rgb_train_x_test = np.array([i for i in rgb_train_x_test]).reshape(-1, 120, 160, 3) / 255.0

    for image in rgb_train_x_test:
        image = image.reshape(1, 120, 160, 3)
        plt.figure()
        plt.imshow(image.squeeze())
        plt.xlabel("Prediction for the {}.frame is {}".format(h, np.argmax(test_model.predict([image]))))
        plt.show()


# model, NAME = create_model()
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs/{}'.format(NAME))

if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'
    sess = tf.Session(config=config)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    keras.backend.set_session(tf.Session(config=config))

    # # # XCEPTION MODEL
    xception_m = xception_model()
    # #
    # # if MODEL_LOAD:
    # #     xception_m.load_weights("cosmo_xception_10-0.1")
    # #     print("model loaded: {}".format("cosmo_xception_10-0.1"))
    #
    # #  TESTING PURPOSES
    # test_xception_model()
    #
    for epoch in range(EPOCHS):
        data_order = [i for i in range(2, 215)]
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
    #
    # # for t in rgb_train_x[:10]:
    # #     plt.figure()
    # #     plt.imshow(t.squeeze())
    # #     plt.show()
    # # x_train = np.array([i[0] for i in training_data]).reshape(-1, 120, 160, 1) / 255.0
    #
            y_train = [i[1] for i in training_data]
    #
            xception_m.fit(np.array(rgb_train_x), np.array(y_train), batch_size=12, epochs=1, validation_split=0.1,
                   callbacks=[tensorboard])
    #
    # #
    # #         train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    # #         model.fit(np.array(x_train), np.array(y_train), batch_size=12, epochs=5, validation_split=0.1,
    # #                   callbacks=[tensorboard])  # if graph is jiggly, size up the batch
    # #
    # #         # model.save(filepath="models/{}".format(NAME))
            xception_m.save(filepath=NEW_NAME)
