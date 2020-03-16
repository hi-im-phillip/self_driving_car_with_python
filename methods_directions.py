from self_driving_car_with_python.direct_keys import PressKey, ReleaseKey, W, A, S, D
import time

wait_time = 0.09


def go_straight():
    ReleaseKey(A)
    ReleaseKey(D)
    PressKey(W)
    time.sleep(0.09)
    ReleaseKey(W)


def go_left():
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(D)
    time.sleep(0.2)
    ReleaseKey(A)


def go_lil_left():
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(D)
    time.sleep(wait_time)
    ReleaseKey(A)


def go_lil_right():
    PressKey(D)
    ReleaseKey(W)
    ReleaseKey(A)
    time.sleep(wait_time)
    ReleaseKey(D)


def go_right():
    PressKey(D)
    ReleaseKey(W)
    ReleaseKey(A)
    time.sleep(0.2)
    ReleaseKey(D)


def slow_down():
    ReleaseKey(W)
    time.sleep(wait_time)


def lil_gas():
    PressKey(W)
    time.sleep(0.01)
    ReleaseKey(W)


def reverse():
    PressKey(S)
    ReleaseKey(S)