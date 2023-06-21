import numpy as np


EPSILON = 10**-9


def normalize(vector):
    return vector / np.linalg.norm(vector)


def perpendicular(vector):
    perpendicular_vector = np.cross(vector, np.array([1, 0, 0]))
    if (perpendicular_vector == 0).all():
        perpendicular_vector = np.cross(vector, np.array([0, 1, 0]))

    return normalize(perpendicular_vector)
