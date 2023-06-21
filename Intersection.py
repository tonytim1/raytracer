from Ray import Ray
from utils import normalize


class Intersection:
    def __init__(self, t, surface, ray: Ray):
        self.t = t
        self.surface = surface
        self.ray = ray
        self.intersect_pos = self.ray.get_position(self.t)

    def get_normal(self):
        # intersect_pos = self.ray.get_position(self.t)
        return normalize(self.surface.calc_normal(self.intersect_pos))
