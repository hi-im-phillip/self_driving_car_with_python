from self_driving_car_with_python.executor import execute
import os
import tensorflow as tf
from tensorflow import keras

# execute choices
# create_data | train_data | run_model | self_driver_cv


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    config.gpu_options.allocator_type = 'BFC'
    config.gpu_options.allow_growth = True
    keras.backend.set_session(tf.Session(config=config))

    execute("run_model")







