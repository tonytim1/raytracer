import numpy as np

# We use epsilon when comparing vectors since they can differ a bit due to rounding
EPSILON = 10**-9


def normalize(vector):
    return vector / np.linalg.norm(vector)


def get_perpendicular_plane(vector):
    if vector[0] != 0:
        return np.cross(vector, np.array([1, 0, 0]))
    elif vector[1] != 0:
        return np.cross(vector, np.array([0, 1, 0]))
    elif vector[2] != 0:
        return np.cross(vector, np.array([0, 0, 1]))
    return None
