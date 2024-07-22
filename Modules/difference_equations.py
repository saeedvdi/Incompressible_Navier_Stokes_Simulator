import numpy as np

# to point to consider
# first, always exclude boundaries from your calcualtion
# f[1:-1, 1:-1] is your u_i_j

class DiscretizationSchemes:
    @staticmethod
    def forward_difference_x(f, dx):
        """Calculate forward difference in x direction."""
        diff = np.zeros_like(f, dtype=np.longdouble)
        diff[1:-1, 1:-1] = (f[1:-1, 2:] - f[1:-1, 1:-1]) / dx
        return diff
    
    @staticmethod
    def forward_difference_y(f, dy):
        """Calculate forward difference in y direction."""
        diff = np.zeros_like(f, dtype=np.longdouble)
        diff[1:-1, 1:-1] = (f[2:, 1:-1] - f[1:-1, 1:-1]) / dy
        return diff
    
    @staticmethod
    def backward_difference_x(f, dx):
        """Calculate backward difference in x direction."""
        diff = np.zeros_like(f, dtype=np.longdouble)
        diff[1:-1, 1:-1] = (f[1:-1, 1:-1] - f[1:-1, :-2]) / dx
        return diff
    
    @staticmethod    
    def backward_difference_y(f, dy):
        """Calculate backward difference in y direction."""
        diff = np.zeros_like(f, dtype=np.longdouble)
        diff[1:-1, 1:-1] = (f[1:-1, 1:-1] - f[:-2, 1:-1]) / dy
        return diff
    
    @staticmethod
    def central_difference_x(f, dx):
        """Calculate central difference in x direction."""
        diff = np.zeros_like(f, dtype=np.longdouble)
        diff[1:-1, 1:-1] = (f[1:-1, 2:] - f[1:-1, :-2]) / (2*dx)
        return diff
    
    @staticmethod    
    def central_difference_y(f, dy):
        """Calculate central difference in y direction."""
        diff = np.zeros_like(f, dtype=np.longdouble)
        diff[1:-1, 1:-1] = (f[2:, 1:-1] - f[:-2, 1:-1]) / (2*dy)
        return diff
    
    @staticmethod
    def upwind_x(f, dx, ux):
        diff = np.zeros_like(f, dtype=np.longdouble)
        if ux >= 0:
            diff[1:-1, 1:-1] = (f[1:-1, 1:-1] - f[1:-1, :-2]) / dx  # Upwind for ux >= 0
        else:
            diff[1:-1, 1:-1] = (f[1:-1, 2:] - f[1:-1, 1:-1]) / dx    # Upwind for ux< 0
        return diff
    
    def upwind_y(f, dy, uy):
            diff = np.zeros_like(f, dtype=np.longdouble)
            if uy >= 0:
                diff[1:-1, 1:-1] = (f[1:-1, 1:-1] - f[:-2, 1:-1]) / dy  # Upwind for uy >= 0
            else:
                diff[1:-1, 1:-1] = (f[2:, 1:-1] - f[1:-1, 1:-1]) / dy    # Upwind for uy < 0
            
            return diff
    
    @staticmethod
    def laplacian(f, dx, dy):
        """Calculate the Laplacian of f."""
        diff = np.zeros_like(f, dtype=np.longdouble)
        diff[1:-1, 1:-1] = (
            (f[1:-1, 2:] - 2*f[1:-1, 1:-1] + f[1:-1, :-2]) / (dx**2) +
            (f[2:, 1:-1] - 2*f[1:-1, 1:-1] + f[:-2, 1:-1]) / (dy**2)
        )
        return diff
    
    @staticmethod
    def second_derivative_x(f, dx):
        """Calculate the second derivative in x direction."""
        diff = np.zeros_like(f, dtype=np.longdouble)
        diff[1:-1, 1:-1] = (f[1:-1, 2:] - 2*f[1:-1, 1:-1] + f[1:-1, :-2]) / (dx**2)
        return diff

    @staticmethod
    def second_derivative_y(f, dy):
        """Calculate the second derivative in y direction."""
        diff = np.zeros_like(f, dtype=np.longdouble)
        diff[1:-1, 1:-1] = (f[2:, 1:-1] - 2*f[1:-1, 1:-1] + f[:-2, 1:-1]) / (dy**2)
        return diff
    
    @staticmethod
    def get_schemes():
        return {
            'forward_difference_x': DiscretizationSchemes.forward_difference_x,
            'forward_difference_y': DiscretizationSchemes.forward_difference_y,
            'backward_difference_x': DiscretizationSchemes.backward_difference_x,
            'backward_difference_y': DiscretizationSchemes.backward_difference_y,
            'central_difference_x': DiscretizationSchemes.central_difference_x,
            'central_difference_y': DiscretizationSchemes.central_difference_y,
            'laplacian': DiscretizationSchemes.laplacian,
            'second_derivative_x': DiscretizationSchemes.second_derivative_x,
            'second_derivative_y': DiscretizationSchemes.second_derivative_y,
        }