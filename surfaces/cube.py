import numpy as np

from utils import EPSILON


class Cube:
    def __init__(self, position, scale, material_index):
        self.position = np.array(position)
        self.scale = scale
        self.material_index = material_index
        self.min_cords = self.position - (self.scale / 2, self.scale / 2, self.scale / 2)
        self.max_cords = self.position + (self.scale / 2, self.scale / 2, self.scale / 2)

    def intersect(self, ray):
        t_min = (self.min_cords[0] - ray.starting_pos[0]) / ray.V[0]
        t_max = (self.max_cords[0] - ray.starting_pos[0]) / ray.V[0]

        if t_min > t_max:
            t_min, t_max = t_max, t_min

        ty_min = (self.min_cords[1] - ray.starting_pos[1]) / ray.V[1]
        ty_max = (self.max_cords[1] - ray.starting_pos[1]) / ray.V[1]

        if ty_min > ty_max:
            ty_min, ty_max = ty_max, ty_min

        if t_min > ty_max or ty_min > t_max:
            return None

        if ty_min > t_min:
            t_min = ty_min

        if ty_max < t_max:
            t_max = ty_max

        tz_min = (self.min_cords[2] - ray.starting_pos[2]) / ray.V[2]
        tz_max = (self.max_cords[2] - ray.starting_pos[2]) / ray.V[2]

        if tz_min > tz_max:
            tz_min, tz_max = tz_max, tz_min

        if t_min > tz_max or tz_min > t_max:
            return None

        if tz_min > t_min:
            t_min = tz_min

        return t_min

    def calc_normal(self, intersect_pos):
        # TODO: do we need EPSILON?
        if abs((intersect_pos[0] - self.position[0]) - self.scale / 2) < EPSILON:
            return np.array((1, 0, 0))

        elif abs((self.position[0] - intersect_pos[0]) - self.scale / 2) < EPSILON:
            return np.array((-1, 0, 0))

        elif abs((intersect_pos[1] - self.position[1]) - self.scale / 2) < EPSILON:
            return np.array((0, 1, 0))

        elif abs((self.position[1] - intersect_pos[1]) - self.scale / 2) < EPSILON:
            return np.array((0, -1, 0))

        elif abs((intersect_pos[2] - self.position[2]) - self.scale / 2) < EPSILON:
            return np.array((0, 0, 1))

        else:
            return np.array((0, 0, -1))
