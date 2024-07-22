import numpy as np
from input_variables import SetupParameters

class MatrixInitializer:
    params = SetupParameters(nx=41, ny=41)
    nx, ny = params.nx, params.ny

    @staticmethod
    def matrix_initialization(nx, ny, zero_initialization=True, initial_value=None):
        if zero_initialization:
            return np.zeros([nx, ny], dtype=np.longdouble)
        elif initial_value is not None:
            return np.full([nx, ny], initial_value, dtype=np.longdouble)
        else:
            raise ValueError("For non-zero initialization, an initial_value must be provided.")
