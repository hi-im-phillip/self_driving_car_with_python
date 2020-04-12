from self_driving_car_with_python.executor import execute
import numpy as np
import pandas as pd
import cv2
# execute choices
# create_data | shuffle_data | train_data | run_model | self_driver_cv
import tensorflow as tf

if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.4
    # session = tf.Session(config=config)

    execute("run_model")
    # training_data = np.load("training_data-1.npy", allow_pickle=True)
    # # print(len(training_data))
    # # df = pd.DataFrame(training_data)
    # # print(df.head())
    # # print(Counter(df[1].apply(str)))
    # for i in training_data:
    #     img = i[0]
    #     print(i[0].shape)
    #     cv2.imshow('window', img)
    #     if cv2.waitKey(10) & 0xFF == ord('q'):
    #         cv2.destroyAllWindows()
    #         break






