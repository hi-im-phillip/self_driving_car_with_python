from self_driving_car_with_python.executor import execute
import tensorflow as tf
# execute choices
# create_data | train_data | run_model | self_driver_cv


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.4
    # session = tf.Session(config=config)

    execute("run_model")







