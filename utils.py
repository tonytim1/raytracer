import numpy as np


EPSILON = 10**-9


def normalize(vector):
    return vector / np.linalg.norm(vector)
