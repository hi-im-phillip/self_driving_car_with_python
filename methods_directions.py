from self_driving_car_with_python.direct_keys import PressKey, ReleaseKey, W, A, S, D
import time
from statistics import mean

t_time = 0.09


def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)


def left():
    PressKey(W)
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(D)
    #ReleaseKey(A)
    time.sleep(t_time)
    ReleaseKey(A)


def right():
    PressKey(W)
    PressKey(D)
    ReleaseKey(W)
    ReleaseKey(A)
    #ReleaseKey(D)
    time.sleep(t_time)
    ReleaseKey(D)


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
    elif 400 >= mean(l2[0::1]) > 0:  # imbalance, both lines are on left side
        go_hard_right()
        print("imbalanced, turning hard right")


def steering_logic(steering_angle, l1, l2):
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


def reverse():
    PressKey(S)
    time.sleep(0.03)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)


def forward_left():
    PressKey(A)
    PressKey(W)
    time.sleep(0.03)
    ReleaseKey(D)
    ReleaseKey(S)
    ReleaseKey(W)


def forward_right():
    PressKey(D)
    PressKey(W)
    time.sleep(0.03)
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(W)


def reverse_left():
    PressKey(S)
    PressKey(A)
    time.sleep(0.03)
    ReleaseKey(W)
    ReleaseKey(D)


def reverse_right():
    PressKey(S)
    PressKey(D)
    time.sleep(0.03)
    ReleaseKey(W)
    ReleaseKey(A)


def no_keys():
    PressKey(W)
    time.sleep(0.03)
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(D)