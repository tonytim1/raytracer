import numpy as np


class InfinitePlane:
    def __init__(self, normal, offset, material_index):
        self.normal = normal
        self.offset = offset
        self.material_index = material_index

    def intersect(self, ray):
        dot = np.dot(ray.V, self.normal)
        if dot == 0:
            return None

        return -1 * (np.dot(ray.starting_position, self.normal) - self.offset) / dot

    def calc_normal(self, intersect_pos):
        return self.normal
