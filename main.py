from self_driving_car_with_python.methods_capturing_frames import make_frames
import  time

if __name__ == '__main__':

    for i in list(range(3))[::-1]:
        print(i + 1)
        time.sleep(1)

    make_frames()