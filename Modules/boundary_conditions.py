import numpy as np
from grid import *

class BCs:
    # tentative step
    def velocity_BCx(ux):
        ux[-1,:] = ux[-2,:] # Top boundary condition, set velocity on cavity lid equal to 1
        ux[0,:] = ux[1,:] # Bottom boundary condition
        ux[:,-1] = ux[:,-2] # Right boundary condition
        ux[:,0] = ux[:,1] # Left boundary condition
        return ux
    
    def velocity_BCy(uy):
        uy[-1,:] = uy[-2,:] # Top boundary condition
        uy[0,:] = uy[1,:] # Bottom boundary condition
        uy[:,-1] = uy[:,-2] # Right boundary condition
        uy[:,0] = uy[:,1] # Left boundary condition
        return uy
    
    def pressure_BC(p):
        p[-1,:] = p[-2,:] # Top boundary condition, p = 0 at y = 2
        p[0,:] = p[1,:] # Bottom Boundary condition, dp/dy = 0 at y = 0
        p[:,0] = p[:, 1] # Left boundary condition, dp/dx = 0 at x = 0
        p[:,-1] = p[:, -2] # Right boundary condition, dp/dx = 0 at x=2
        return p
    
    def wall_pressure_B_x(p):
        p[:, 0] = p[:, 1]  # dp/dx = 0 at x = 0 # Left boundary condition
        p[:, -1] = p[:, -2]  # dp/dx = 0 at x = 2 # Right boundary condition
        
    def wall_pressure_B_y(p):
        p[-1, :] =p[-2, :]  # dp/dy = 0 at y = 2
        p[0, :] = p[1, :]  # dp/dy = 0 at y = 0
        
    def temperature_BC(T):
        # T[-1,:] = T[-2,:] # Top boundary condition 
        # T[0,:] = 300 # Bottom Boundary condition
        # T[:,0] = T[:,1]  # Left boundary condition
        # T[:,-1] = T[:,-2] # Right boundary condition
        T[-1,:] = 0 # dp/dy = 0 at y = 2 # Top boundary condition 
        T[0,:] = T[1,:] # dp/dy = 0 at y = 0 # Bottom Boundary condition
        T[:,0] = T[:,1]  # dp/dx = 0 at x = 0 # Left boundary condition
        T[:,-1] = T[:,-2] # dp/dx = 0 at x = 2 # Right boundary condition
        return T

class SourceBC: 
    @staticmethod   
    def source_temp_BC(T, source_temp,source_radius, lx, ly, nx, ny):
        source_center_x, source_center_y = lx / 2, ly / 2
        X, Y, dx, dy = MeshGrid.mesh(nx, ny, lx, ly)
        for i in range(ny):
            for j in range(nx):
                if ((X[i, j] - source_center_x)**2 + (Y[i, j] - source_center_y)**2) <= source_radius**2:
                    T[i, j] = source_temp
        return T
    
    @staticmethod
    def source_ux_BC(ux, source_ux, source_radius, lx, ly, nx, ny):
        source_center_x, source_center_y = lx / 2, ly / 2
        X, Y, dx, dy = MeshGrid.mesh(nx, ny, lx, ly)
        for i in range(ny):
            for j in range(nx):
                if ((X[i, j] - source_center_x)**2 + (Y[i, j] - source_center_y)**2) <= source_radius**2:
                    ux[i, j] = source_ux
        return ux
    
    @staticmethod
    def source_uy_BC(uy, source_uy, source_radius, lx, ly, nx, ny):
        source_center_x, source_center_y = lx / 2, ly / 2
        X, Y, dx, dy = MeshGrid.mesh(nx, ny, lx, ly)
        for i in range(ny):
            for j in range(nx):
                if ((X[i, j] - source_center_x)**2 + (Y[i, j] - source_center_y)**2) <= source_radius**2:
                    uy[i, j] = source_uy
        return uy
    
    # Cable number 2
    @staticmethod   
    def source_temp_BC_2(T, source_temp,source_radius, lx, ly, nx, ny):
        source_center_x, source_center_y = lx / 2, ly / 2
        X, Y, dx, dy = MeshGrid.mesh(nx, ny, lx, ly)
        for i in range(ny):
            for j in range(nx):
                if ((X[i, j] - source_center_x - 0.1)**2 + (Y[i, j] - source_center_y)**2) <= source_radius**2:
                    T[i, j] = source_temp
        return T
    
    @staticmethod
    def source_ux_BC_2(ux, source_ux, source_radius, lx, ly, nx, ny):
        source_center_x, source_center_y = lx / 2, ly / 2
        X, Y, dx, dy = MeshGrid.mesh(nx, ny, lx, ly)
        for i in range(ny):
            for j in range(nx):
                if ((X[i, j] - source_center_x - 0.1)**2 + (Y[i, j] - source_center_y)**2) <= source_radius**2:
                    ux[i, j] = source_ux
        return ux
    
    @staticmethod
    def source_uy_BC_2(uy, source_uy, source_radius, lx, ly, nx, ny):
        source_center_x, source_center_y = lx / 2, ly / 2
        X, Y, dx, dy = MeshGrid.mesh(nx, ny, lx, ly)
        for i in range(ny):
            for j in range(nx):
                if ((X[i, j] - source_center_x - 0.1)**2 + (Y[i, j] - source_center_y)**2) <= source_radius**2:
                    uy[i, j] = source_uy
        return uy
    
    # cable number 3
    @staticmethod   
    def source_temp_BC_3(T, source_temp,source_radius, lx, ly, nx, ny):
        source_center_x, source_center_y = lx / 2, ly / 2
        X, Y, dx, dy = MeshGrid.mesh(nx, ny, lx, ly)
        for i in range(ny):
            for j in range(nx):
                if ((X[i, j] - source_center_x + 0.1)**2 + (Y[i, j] - source_center_y)**2) <= source_radius**2:
                    T[i, j] = source_temp
        return T
    
    @staticmethod
    def source_ux_BC_3(ux, source_ux, source_radius, lx, ly, nx, ny):
        source_center_x, source_center_y = lx / 2, ly / 2
        X, Y, dx, dy = MeshGrid.mesh(nx, ny, lx, ly)
        for i in range(ny):
            for j in range(nx):
                if ((X[i, j] - source_center_x + 0.1)**2 + (Y[i, j] - source_center_y)**2) <= source_radius**2:
                    ux[i, j] = source_ux
        return ux
    
    @staticmethod
    def source_uy_BC_3(uy, source_uy, source_radius, lx, ly, nx, ny):
        source_center_x, source_center_y = lx / 2, ly / 2
        X, Y, dx, dy = MeshGrid.mesh(nx, ny, lx, ly)
        for i in range(ny):
            for j in range(nx):
                if ((X[i, j] - source_center_x + 0.1)**2 + (Y[i, j] - source_center_y)**2) <= source_radius**2:
                    uy[i, j] = source_uy
        return uy