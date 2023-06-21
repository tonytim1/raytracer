
class Ray:
    def __init__(self, starting_position, V):
        self.starting_position = starting_position
        self.V = V

    def get_position(self, t):
        return self.starting_position + t * self.V
