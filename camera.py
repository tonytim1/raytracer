from utils import normalize


class Camera:
    def __init__(self, position, look_at, up_vector, screen_distance, screen_width):
        self.position = position
        self.look_at = look_at
        self.up_vector = up_vector
        self.screen_distance = screen_distance
        self.screen_width = screen_width
        self.screen_height = None
        self.towards = normalize(look_at - position)

    def set_screen_height(self, width, height):
        self.screen_height = (height / width) * self.screen_width
