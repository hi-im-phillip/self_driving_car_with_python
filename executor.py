from self_driving_car_with_python.execute_functions import *
import logging


def execute(choice):

    while True:
        if choice == "create_data":
            print("Creating data for CNN")
            execute_create_training_data()
            break
        elif choice == "shuffle_data":
            print("Shuffle data for CNN")
            execute_shuffle_data()
            break
        elif choice == "train_data":
            print("Training data for CNN")
            execute_to_train_data()
            break
        elif choice == "run_model":
            print("Running model")
            execute_model_new()
            break
        elif choice == "self_driver_cv":
            print("Running self driving driver cv")
            execute_self_driving_cv()
            break
        else:
            print(
                "Wrong input. \n Try with: \n 1.create_data \n 2.shuffle_data \n 3.train_data \n 4.run_model "
                "\n 5.self_driver_cv")
            time.sleep(1000)