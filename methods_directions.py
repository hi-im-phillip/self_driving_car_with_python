from self_driving_car_with_python.direct_keys import PressKey, ReleaseKey, W, A, S, D
from self_driving_car_with_python.get_keys import w, a, s, d, wd, wa, sd, sa, nk
from self_driving_car_with_python.ProjectCarsAPI import get_current_car_speed
import time
from statistics import mean
import numpy as np

t_time = 0.09

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
    print(car_speed)
    if np.argmax(prediction) == np.argmax(w):
        straight_v2()
        print("straight")
    elif np.argmax(prediction) == np.argmax(s):
        if car_speed <= 5:
            print("Speed was {} but said to reverse it".format(str(car_speed)))
            straight_v2()
        else:
            reverse_v2()
            print("reverse")
    elif np.argmax(prediction) == np.argmax(a):
        if car_speed > 35:
            print("slowing down and left")
            slowing_down()
            left_v2()
        else:
            left_v2()
            print("left")
    elif np.argmax(prediction) == np.argmax(d):
        if car_speed > 35:
            print("slowing down and right")
            slowing_down()
            right_v2()
        else:
            right_v2()
            print("right")
    elif np.argmax(prediction) == np.argmax(wa):
        forward_left_v2()
        print("forward_left")
    elif np.argmax(prediction) == np.argmax(wd):
        forward_right_v2()
        print("forward_right")
    elif np.argmax(prediction) == np.argmax(sa):
        reverse_left_v2()
        print("reverse_left")
    elif np.argmax(prediction) == np.argmax(sd):
        reverse_right_v2()
        print("reverse_right")
    elif np.argmax(prediction) == np.argmax(nk):
        no_keys()
        print("no_keys")


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


def forward_left():
    PressKey(A)
    PressKey(W)
    time.sleep(0.02)
    ReleaseKey(D)
    ReleaseKey(S)
    # ReleaseKey(W)
    # ReleaseKey(A)


def forward_right():
    PressKey(D)
    PressKey(W)
    time.sleep(0.02)
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(W)
    ReleaseKey(D)


def reverse_left():
    PressKey(S)
    PressKey(A)
    time.sleep(0.03)
    ReleaseKey(W)
    ReleaseKey(D)
    ReleaseKey(S)
    ReleaseKey(A)


def reverse_right():
    PressKey(S)
    PressKey(D)
    time.sleep(0.03)
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


def left_v2():
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(D)
    ReleaseKey(S)
    # time.sleep(0.2)
    # ReleaseKey(A)


def right_v2():
    PressKey(D)
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    # time.sleep(0.2)
    # ReleaseKey(D)


def reverse_v2():
    PressKey(S)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)


def forward_left_v2():
    PressKey(W)
    PressKey(A)
    ReleaseKey(D)
    ReleaseKey(S)
    time.sleep(0.2)
    ReleaseKey(A)
    # ReleaseKey(W)


def forward_right_v2():
    PressKey(W)
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)
    time.sleep(0.2)
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

