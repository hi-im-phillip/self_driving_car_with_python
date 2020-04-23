import time


def timer(seconds=3):
    for i in list(range(seconds))[::-1]:
        print(i + 1)
        time.sleep(1)


def show_turn_and_speed(turn, speed, show_speed=True):
    if show_speed:
        print("Turn: {} \nSpeed: {}".format(str(turn), str(speed)))
    else:
        print("Turn: {}".format(str(turn)))

