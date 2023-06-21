from ctypes import Union

from Ray import Ray
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere
from utils import normalize


class Intersection:
    def __init__(self, t, obj: Union[Cube, Sphere, InfinitePlane], ray: Ray):
        self.t = t
        self.obj = obj
        self.ray = ray
        self.intersect_pos = self.ray.get_position(self.t)

    def get_normal(self):
        # intersect_pos = self.ray.get_position(self.t)
        return normalize(self.obj.calc_normal(self.intersect_pos))
