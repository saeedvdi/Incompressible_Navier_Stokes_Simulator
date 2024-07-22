import numpy as np

class NavierStokesInitial:
    @staticmethod
    def set_initial_field(shape, initial_condition):
        """
        Set the initial field for Navier-Stokes simulation.

        Parameters:
        shape (tuple): Shape of the field (e.g., (nx,) for 1D or (ny, nx) for 2D)
        initial_condition: Can be a scalar, array-like, or callable function

        Returns:
        numpy.ndarray: Array representing the initial field
        """
        if callable(initial_condition):
            return initial_condition(*map(np.arange, shape))
        elif np.isscalar(initial_condition):
            return np.full(shape, initial_condition)
        else:
            return np.array(initial_condition)

    @classmethod
    def initial_velocity_x_field(cls, nx, initial_condition):
        """
        Create the initial x-velocity field.

        Parameters:
        nx (int): Number of spatial points in x direction
        initial_condition: Scalar, array-like of length nx, or function f(x)

        Returns:
        numpy.ndarray: 1D array representing the initial x-velocity field
        """
        return cls.set_initial_field((nx,), initial_condition)

    @classmethod
    def initial_velocity_y_field(cls, ny, initial_condition):
        """
        Create the initial y-velocity field.

        Parameters:
        ny (int): Number of spatial points in y direction
        initial_condition: Scalar, array-like of length ny, or function f(y)

        Returns:
        numpy.ndarray: 1D array representing the initial y-velocity field
        """
        return cls.set_initial_field((ny,), initial_condition)

    @classmethod
    def initial_pressure_field(cls, nx, ny, initial_condition):
        """
        Create the initial pressure field.

        Parameters:
        nx (int): Number of spatial points in x direction
        ny (int): Number of spatial points in y direction
        initial_condition: Scalar, array-like of shape (ny, nx), or function f(x, y)

        Returns:
        numpy.ndarray: 2D array representing the initial pressure field
        """
        return cls.set_initial_field((ny, nx), initial_condition)