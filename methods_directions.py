from self_driving_car_with_python.direct_keys import PressKey, ReleaseKey, W, A, S, D
from self_driving_car_with_python.get_keys import w, a, s, d, wd, wa, sd, sa, nk
from self_driving_car_with_python.ProjectCarsAPI import get_current_car_speed, get_current_time_game, get_steering_angle
from self_driving_car_with_python.methods_helper import show_turn_and_speed
import time
from statistics import mean
import numpy as np
from random import randrange


# TWEAKS FOR CV


def go_straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    time.sleep(0.2)
    ReleaseKey(W)


def go_left():
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(D)
    time.sleep(0.24)
    ReleaseKey(A)


def go_right():
    PressKey(D)
    ReleaseKey(W)
    ReleaseKey(A)
    time.sleep(0.24)
    ReleaseKey(D)


def go_lil_left():
    ReleaseKey(D)
    PressKey(A)
    time.sleep(0.17)
    ReleaseKey(A)


def go_lil_right():
    ReleaseKey(A)
    PressKey(D)
    time.sleep(0.17)
    ReleaseKey(D)


def slow_down():
    ReleaseKey(W)
    time.sleep(0.09)


def lil_gas():
    PressKey(W)
    time.sleep(0.06)
    ReleaseKey(W)


def reverse():
    PressKey(S)
    ReleaseKey(S)


def go_hard_left():
    # ReleaseKey(W)
    # ReleaseKey(A)
    # PressKey(D)
    # time.sleep(0.1)
    # ReleaseKey(D)
    ReleaseKey(W)
    ReleaseKey(D)
    PressKey(A)
    time.sleep(0.35)
    ReleaseKey(A)


def go_hard_right():
    # ReleaseKey(W)
    # ReleaseKey(D)
    # PressKey(A)
    # time.sleep(0.1)
    # ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(A)
    PressKey(D)
    time.sleep(0.35)
    ReleaseKey(D)


def balance_car(l1, l2):
    if mean(l1[0::1]) >= 450:  # imbalance, both lines are on right side
        go_hard_left()
        print("imbalanced, turning hard left")
    elif 420 >= mean(l2[0::1]) > 0:  # imbalance, both lines are on left side
        go_hard_right()
        print("imbalanced, turning hard right")


def steering_logic(steering_angle, l1, l2):
    """
    Steering logic for self_driving_cv
    """
    balance_car(l1, l2)

    if steering_angle < 85:
        if steering_angle < 60:
            if steering_angle < 20:
                go_hard_left()
                print("going hard left")
                print(steering_angle)
            else:
                go_left()
                # lil_gas()
                print("going left")
                print(steering_angle)
        else:
            go_lil_left()
            lil_gas()
            print("going lil left")
            print(steering_angle)
    elif steering_angle > 95:
        if steering_angle > 120:
            if steering_angle > 160:
                go_hard_right()
                print("going hard right")
                print(steering_angle)
            else:
                go_right()
                # lil_gas()
                print("going right")
                print(steering_angle)
        else:
            go_lil_right()
            lil_gas()
            print("going lil right")
            print(steering_angle)
    else:
        go_straight()
        print("going straight")
        print(steering_angle)


# TWEAKS FOR NN


def make_direction_logic(prediction):
    car_speed = get_current_car_speed()
    steering_angle = get_steering_angle()
    if np.argmax(prediction) == np.argmax(w):  # Forward
        straight_v2()
        # show_turn_and_speed("Forward", car_speed, show_speed=False)
    elif np.argmax(prediction) == np.argmax(s):  # Reverse
        if car_speed <= 15.0:
            straight_v2()
            # time.sleep(1)
            print("Speed was too low for reverse {}".format(str(car_speed)))
        else:
            reverse_v2()
            # show_turn_and_speed("Reverse", car_speed, show_speed=False)
    elif np.argmax(prediction) == np.argmax(a):  # Left
        if car_speed <= 0.5 and get_current_time_game() > 10:  # If it's start of the race and needs to take left
            unstuck()
            print("left unstuck")
        else:
            left_v2(steering_angle, car_speed)
            show_turn_and_speed("Left", car_speed, show_speed=True)
    elif np.argmax(prediction) == np.argmax(d):  # Right
        if car_speed <= 0.5 and get_current_time_game() > 10:
            unstuck()
            print("right unstuck")
        else:
            right_v2(steering_angle, car_speed)
            show_turn_and_speed("Right", car_speed, show_speed=True)
    elif np.argmax(prediction) == np.argmax(wa):  # Forward Left
        forward_left(steering_angle, car_speed)
        # show_turn_and_speed("Forward Left", car_speed, show_speed=False)
    elif np.argmax(prediction) == np.argmax(wd):  # Forward Right
        forward_right(steering_angle, car_speed)
        # show_turn_and_speed("Forward Right", car_speed, show_speed=False)
    elif np.argmax(prediction) == np.argmax(sa):  # Reverse Left
        reverse_left_v2()
        # show_turn_and_speed("Reverse Left", car_speed, show_speed=False)
    elif np.argmax(prediction) == np.argmax(sd):  # Reverse Right
        reverse_right_v2()
        # show_turn_and_speed("Reverse Right", car_speed, show_speed=False)
    elif np.argmax(prediction) == np.argmax(nk):  # No Keys
        random_num = randrange(0, 1)
        if random_num == 0:
            no_keys()
        elif random_num == 1:
            PressKey(W)
        if car_speed <= 0.5 and get_current_time_game() > 10:
            unstuck()
            print("right unstuck")
        # show_turn_and_speed("No Keys", car_speed, show_speed=False)


def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)
    time.sleep(0.13)
    # ReleaseKey(W)


def left():
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(D)
    ReleaseKey(S)
    time.sleep(0.06)
    # ReleaseKey(A)


def right():
    PressKey(D)
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    time.sleep(0.06)
    # ReleaseKey(D)


def reverse():
    PressKey(S)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)
    time.sleep(0.05)
    ReleaseKey(S)


def forward_left(steering_angle, car_speed):
    PressKey(A)
    PressKey(W)
    ReleaseKey(D)
    ReleaseKey(S)
    if car_speed > 30 and car_speed < 58:
        PressKey(S)
        ReleaseKey(S)
        if steering_angle < -0.12 or steering_angle > 0.12:
            ReleaseKey(W)
            ReleaseKey(A)
            print("Angle was less -0.12 or more 0.12")
    elif car_speed > 58:
        if steering_angle < -0.08 or steering_angle > 0.08:
            ReleaseKey(A)
            print("Angle was less -0.08 or more 0.08")
    elif steering_angle < -0.16 or steering_angle > 0.16:
        ReleaseKey(W)
        ReleaseKey(A)
        print("Angle was less -0.16 or more 0.16")


def forward_right(steering_angle, car_speed):
    PressKey(D)
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    if car_speed > 30 and car_speed < 58:
        PressKey(S)
        ReleaseKey(S)
        if steering_angle < -0.12 or steering_angle > 0.12:
            ReleaseKey(W)
            ReleaseKey(D)
            print("Angle was less -0.12 or more 0.12")
    elif car_speed > 58:
        if steering_angle < -0.08 or steering_angle > 0.08:
            ReleaseKey(D)
            print("Angle was less -0.08 or more 0.08")
    elif steering_angle < -0.16 or steering_angle > 0.16:
        ReleaseKey(W)
        ReleaseKey(D)
        print("Angle was less -0.16 or more 0.16")


def reverse_left():
    PressKey(S)
    PressKey(A)
    time.sleep(0.02)
    ReleaseKey(W)
    ReleaseKey(D)
    ReleaseKey(S)
    ReleaseKey(A)


def reverse_right():
    PressKey(S)
    PressKey(D)
    time.sleep(0.02)
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(D)


def no_keys():
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(D)
    ReleaseKey(W)


def straight_v2():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)


def left_v2(steering_angle=None, car_speed=None):
    PressKey(A)
    ReleaseKey(D)
    ReleaseKey(W)
    ReleaseKey(S)
    if car_speed >= 20.0:
        PressKey(S)
        if steering_angle < -0.27 or steering_angle > 0.27:
            ReleaseKey(A)
        print("slowing down and left")
    elif car_speed > 5 and car_speed < 15:
        if steering_angle < -0.40 or steering_angle > 0.40:
            ReleaseKey(A)
            PressKey(W)
            # ReleaseKey(W)
            print("Angle was more 0.40 or less -0.40")
    elif steering_angle < -0.45 or steering_angle > 0.45:
        ReleaseKey(A)
        print("Angle was more 0.45 or less -0.45")


def right_v2(steering_angle=None, car_speed=None):
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(S)
    if car_speed >= 20.0:
        PressKey(S)
        if steering_angle < -0.30 or steering_angle > 0.30:
            ReleaseKey(D)
        print("slowing down and right")
    elif car_speed > 5 and car_speed < 17:
        if steering_angle < -0.35 or steering_angle > 0.35:
            ReleaseKey(D)
            PressKey(W)
            # ReleaseKey(W)
            print("Angle was more 0.40 or less -0.40")
    elif steering_angle < -0.40 or steering_angle > 0.40:
        ReleaseKey(D)
        print("Angle was more 0.45 or less -0.45")


def reverse_v2():
    PressKey(S)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)


def forward_left_v2(time_sleep=0):
    PressKey(W)
    PressKey(A)
    ReleaseKey(D)
    ReleaseKey(S)
    time.sleep(time_sleep)
    ReleaseKey(A)
    # ReleaseKey(W)


def forward_right_v2(time_sleep=0):
    PressKey(W)
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)
    time.sleep(time_sleep)
    ReleaseKey(D)
    # ReleaseKey(W)


def reverse_left_v2():
    PressKey(S)
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(D)


def reverse_right_v2():
    PressKey(S)
    PressKey(D)
    ReleaseKey(W)
    ReleaseKey(A)


def slowing_down():
    PressKey(S)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)
    time.sleep(0.6)
    ReleaseKey(S)


def slowing_down_right():
    PressKey(S)
    PressKey(D)
    ReleaseKey(W)
    ReleaseKey(A)
    time.sleep(0.4)
    # PressKey(D)
    # time.sleep(0.1)


def slowing_down_left():
    PressKey(S)
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(D)
    time.sleep(0.4)
    # PressKey(A)
    # time.sleep(0.1)


def unstuck():
    print("Got Stuck! Trying to get out.")
    no_keys()
    random_number = randrange(0, 3)
    if random_number == 0:
        reverse_v2()
        time.sleep(2)
        straight_v2()
    elif random_number == 1:
        reverse_left_v2()
        time.sleep(2)
        forward_right_v2(2)
    elif random_number == 2:
        reverse_right_v2()
        time.sleep(2)
        forward_left_v2(2)
    else:
        straight_v2()
        time.sleep(2)
