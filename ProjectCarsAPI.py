"""
The MIT License (MIT)

Copyright (c) 2015 Mats Lindh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import carseour
import time

# Project Cars API
# Tracking the game speed in real time

try:
    game = carseour.live()
except carseour.InvalidSharedMemoryVersionException as e:
    pass


def show_speed():
    while True:
        print("Speed: " + str(round(game.mSpeed, 1)) + " m/s")
        time.sleep(0.5)


def get_current_car_speed():
    return game.mSpeed


def get_steering_angle():  # From Left -1 to Right 1
    return game.mSteering


def get_current_time_game():
    return game.mCurrentTime
