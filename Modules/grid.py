import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class MeshGrid(ABC):
    @staticmethod
    @abstractmethod
    def mesh(nx, ny, lx, ly):
        """
        Create a 2D mesh grid.

        Parameters:
        nx (int): Number of points in x direction
        ny (int): Number of points in y direction
        lx (float): Length in x direction
        ly (float): Length in y direction

        Returns:
        tuple: X, Y meshgrid arrays, dx, and dy
        """
        dx = lx / (nx - 1)  # element length in x direction
        dy = ly / (ny - 1)  # element length in y direction
        
        x = np.linspace(0, lx, nx)  # range in x direction
        y = np.linspace(0, ly, ny)  # range in y direction
        
        X, Y = np.meshgrid(x, y)
        return X, Y, dx, dy

    @staticmethod
    def mesh_plot(X, Y):
        """
        Plot the 2D mesh grid.

        Parameters:
        X (numpy.ndarray): X coordinates of the mesh
        Y (numpy.ndarray): Y coordinates of the mesh
        """
        plt.figure(figsize=(8, 6), dpi=100)
        plt.plot(X, Y, color='g', marker='o', markersize=4, linestyle='-')
        plt.plot(X.T, Y.T, color='g', linestyle='-')
        plt.xlabel('x direction', fontsize=12)
        plt.ylabel('y direction', fontsize=12)
        plt.title('Discretized domain in 2D', fontsize=14)
        plt.show()
