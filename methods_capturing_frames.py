from self_driving_car_with_python.grab_screen import grab_screen

screen_height = 1024
screen_width = 768


def take_frame():
    """
    Take a frame of the screen and store it in memory
    region - Specifies specific region (bbox = x cord, y cord, width, height)
    """
    frame = grab_screen(region=(0, 0, screen_height, screen_width))
    return frame





