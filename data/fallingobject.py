import os

import numpy as np


def load_data() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Load the falling object data. The data is a 1D array of the
    object's height at each time step. For convenience, the time and
    time step are also returned.

    Returns:
        (data, t_eval, dt): A tuple containing the data, time stamps,
        and time step.
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data = np.load(dir_path + '/falling_object.npy')

    # Construct the time stamps.
    dt = 0.001
    t_eval = np.arange(0, 1, dt)
    return data, t_eval, dt
