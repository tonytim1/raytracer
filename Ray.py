
class Ray:
    def __init__(self, starting_pos, V):
        self.starting_pos = starting_pos
        self.V = V

    def get_position(self, t):
        return self.starting_pos + t * self.V
