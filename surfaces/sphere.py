from math import sqrt
import numpy as np

from utils import normalize


class Sphere:
    def __init__(self, position, radius, material_index):
        self.position = position
        self.radius = radius
        self.material_index = material_index

    def intersect(self, ray):
        L = self.position - ray.starting_position
        t_ca = np.dot(L, ray.v)
        if t_ca < 0:
            return None

        d_square = np.dot(L, L) - t_ca ** 2
        if d_square > self.radius ** 2:
            return None

        r_squared = self.radius ** 2
        t_hc = sqrt(r_squared - d_square)  # faster than ** 0.5
        return min(t_ca - t_hc, t_ca + t_hc)

    def calc_normal(self, intersect_pos):
        return normalize(intersect_pos - self.position)
