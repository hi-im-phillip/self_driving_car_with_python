import time


def timer():
    for i in list(range(3))[::-1]:
        print(i + 1)
        time.sleep(1)