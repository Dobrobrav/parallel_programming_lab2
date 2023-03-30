from typing import Final

import numpy as np

RELIEF: Final = np.array([
    [-2, -1, 0],
    [-1, 1, 1],
    [0, 1, 2],
], dtype='int16')

BLUR: Final = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
], dtype='int16')

SHOW_EDGES: Final = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0],
], dtype='int16')

INCREASE_CONTRAST: Final = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0],
], dtype='int16')
