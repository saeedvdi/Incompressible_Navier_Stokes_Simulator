import numpy as np
from tqdm import tqdm
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from matrix_init import MatrixInitializer
from boundary_conditions import *
# from input_variables import SetupParameters
# This code has been written using forward Euler method time discretization

class HomogeneousAdvection:
    # params = SetupParameters(nx=41, ny=41)
    # nx, ny, rho_fluid = params.nx, params.ny
    
    @staticmethod
    def advection_viscous_dissipation_temperature(T: np.ndarray, ux: np.ndarray, uy: np.ndarray, dT__dx: np.ndarray, 
                                                  dT__dy: np.ndarray, laplace_T: np.ndarray, nu: float, dt: float) -> np.ndarray:
        T_new = T + dt * (-(ux * dT__dx + uy * dT__dy) + nu * laplace_T)
        return T_new
    
    @staticmethod 
    def advection_diffusion_temperature(T:np.ndarray, ux: np.ndarray, uy:np.ndarray, dT__dx: np.ndarray, dT__dy: np.ndarray,
                                        laplace_T:np.ndarray, alpha:float, dx:int, dy:int, dt:float) -> np.ndarray:
        T_new = T + dt * (-(ux * dT__dx + uy * dT__dy) + alpha * laplace_T)
        return T_new
    
    @staticmethod
    def homogenous_advection_horizontal(ux: np.ndarray, uy: np.ndarray, dux__dx: np.ndarray, dux__dy: np.ndarray, 
                                        laplace__ux: np.ndarray, nu: float, dt: float) -> np.ndarray:
        ux_new = ux + dt * (-(ux * dux__dx + uy * dux__dy) + nu * laplace__ux)
        return ux_new
    
    @staticmethod
    def homogenous_advection_vertical(ux: np.ndarray, uy: np.ndarray, duy__dx: np.ndarray, duy__dy: np.ndarray, 
                                      laplace__uy: np.ndarray, nu: float, dt: float) -> np.ndarray:
        uy_new = uy + dt * (-(ux * duy__dx + uy * duy__dy) + nu * laplace__uy)
        return uy_new
    
class DarcyVelocity:
    @staticmethod 
    # def advection_diffusion_temperature(T:np.ndarray, ux: np.ndarray, uy:np.ndarray, dT__dx: np.ndarray, dT__dy: np.ndarray,
    #                                     laplace_T:np.ndarray, rho_fluid:float, c_p_f:float, lambda_b:float,
    #                                     alpha:float, dx:int, dy:int, dt:float) -> np.ndarray:
    #     # T_new = T + ((dt/(rho_fluid * c_p_f)) * ((lambda_b * laplace_T)-(rho_fluid * c_p_f *(ux * dT__dx + uy * dT__dy)) 
    #     #                                          + alpha * laplace_T))
        
    def advection_diffusion_temperature (
            T:np.ndarray, ux:np.ndarray, uy:np.ndarray, delta_T__dx: np.ndarray, delta_T__dy:np.ndarray,
             laplace_T:np.ndarray, lambda_b:float, rho_fluid:float, c_p_f:float, Q_in:float
            ):    
        T_new = T + rho_fluid * c_p_f * (lambda_b * laplace_T - rho_fluid * c_p_f * (ux * delta_T__dx + uy * delta_T__dy) + Q_in)
        return T_new
        
    @staticmethod
    def darcy_velocity_x(T_new:np.ndarray, T_ref:np.ndarray, 
                         dp__dx:np.ndarray, darcy_ux_old:np.ndarray,
                         rho_fluid:float, beta:float, k:float, n:float, nu:float, dt:float):
        rho = rho_fluid * (1 - beta * (T_new - T_ref))
        darcy_ux = darcy_ux_old + dt * ((-k/(n * nu))*(dp__dx))
        return darcy_ux
        
    def darcy_velocity_y(T_new:np.ndarray, T_ref:np.ndarray, 
                         dp__dy:np.ndarray, darcy_uy_old:np.ndarray,
                         rho_fluid:float, beta:float, k:float, n:float, nu:float, g:float, dt:float):
        rho = rho_fluid * (1 - beta * (T_new - T_ref))
        darcy_uy = darcy_uy_old + dt * ((-k/(n * nu))*(dp__dy + (g * rho)))
        return darcy_uy
    
    @staticmethod
    def darcy_velocity_upwind_x(T_new: np.ndarray, T_ref: np.ndarray, 
                        dp__dx: np.ndarray, ux_old: np.ndarray, uy_old: np.ndarray,
                        rho_fluid: float, beta: float, k: float, 
                        n: float, nu: float, dx: float, dt: float) -> np.ndarray:
        rho = rho_fluid * (1 - beta * (T_new - T_ref))
        
        # Upwind scheme for x-direction
        ux_upwind = np.where(ux_old > 0, 
                            (ux_old - np.roll(ux_old, 1, axis=0)) / dx,
                            (np.roll(ux_old, -1, axis=0) - ux_old) / dx)
        
        uy_upwind = np.where(uy_old > 0,
                            (ux_old - np.roll(ux_old, 1, axis=1)) / dx,
                            (np.roll(ux_old, -1, axis=1) - ux_old) / dx)
        
        ux_new = ux_old + dt * ((-k/(n * nu)) * (dp__dx) - 
                                (ux_old * ux_upwind + uy_old * uy_upwind))
        return ux_new

    @staticmethod
    def darcy_velocity_upwind_y(T_new: np.ndarray, T_ref: np.ndarray, 
                        dp__dy: np.ndarray, ux_old: np.ndarray, uy_old: np.ndarray,
                        rho_fluid: float, beta: float, k: float, 
                        n: float, nu: float, g: float, dy: float, dt: float) -> np.ndarray:
        rho = rho_fluid * (1 - beta * (T_new - T_ref))
        
        # Upwind scheme for y-direction
        ux_upwind = np.where(ux_old > 0,
                            (uy_old - np.roll(uy_old, 1, axis=0)) / dy,
                            (np.roll(uy_old, -1, axis=0) - uy_old) / dy)
        
        uy_upwind = np.where(uy_old > 0,
                            (uy_old - np.roll(uy_old, 1, axis=1)) / dy,
                            (np.roll(uy_old, -1, axis=1) - uy_old) / dy)
        
        uy_new = uy_old + dt * ((-k/(n * nu)) * (dp__dy + (g * rho)) - 
                                (ux_old * ux_upwind + uy_old * uy_upwind))
        return uy_new
    
    @staticmethod
    def implicit_advection_diffusion_temperature(T: np.ndarray, ux: np.ndarray, uy: np.ndarray,
                                                 rho_fluid: float, c_p_f: float, lambda_b: float,
                                                 alpha: float, dx: float, dy: float, dt: float) -> np.ndarray:
        nx, ny = T.shape
        N = nx * ny

        # Create coefficient matrix
        main_diag = np.ones(N) * (1 + 2*dt*(lambda_b/(rho_fluid*c_p_f))*(1/dx**2 + 1/dy**2))
        off_diag_x = np.ones(N-1) * (-dt*(lambda_b/(rho_fluid*c_p_f))/dx**2)
        off_diag_y = np.ones(N-nx) * (-dt*(lambda_b/(rho_fluid*c_p_f))/dy**2)

        diagonals = [main_diag, off_diag_x, off_diag_x, off_diag_y, off_diag_y]
        offsets = [0, -1, 1, -nx, nx]
        A = diags(diagonals, offsets, shape=(N, N), format='csr')

        # Create right-hand side
        b = T.flatten()

        # Add advection terms (using upwind scheme)
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                idx = i*ny + j
                if ux[i,j] > 0:
                    b[idx] -= dt * ux[i,j] * (T[i,j] - T[i-1,j]) / dx
                else:
                    b[idx] -= dt * ux[i,j] * (T[i+1,j] - T[i,j]) / dx
                if uy[i,j] > 0:
                    b[idx] -= dt * uy[i,j] * (T[i,j] - T[i,j-1]) / dy
                else:
                    b[idx] -= dt * uy[i,j] * (T[i,j+1] - T[i,j]) / dy

        # Solve the system
        T_new = spsolve(A, b).reshape((nx, ny))

        return T_new    
    
class PressurePoissonGaussSeidel:
    @staticmethod
    def source_term_gauss_seidel(ux: np.ndarray, uy: np.ndarray, dx: float, dy: float, dt: float) -> np.ndarray:
        # Initialize b_i_j with zeros
        b_i_j = np.zeros_like(ux)
        b_i_j[1:-1, 1:-1] = (1/dt) * (
            (ux[1:-1, 2:] - ux[1:-1, :-2]) / (2 * dx) +
            (uy[2:, 1:-1] - uy[:-2, 1:-1]) / (2 * dy)
        )
        return b_i_j
    
    @staticmethod
    def pressure_poisson_gauss_seidel(p_prev: np.ndarray, b_i_j: np.ndarray, nppi: int, dx: float, dy: float) -> np.ndarray:
        p_new = p_prev.copy()
        # for _ in range(nppi):
        #     p_prev = p_new.copy()
        #     for i in range(1, p_new.shape[0] - 1):
        #         for j in range(1, p_new.shape[1] - 1):
        #             p_new[i, j] = (
        #                 (dy**2 * (p_new[i+1, j] + p_new[i-1, j]) +
        #                  dx**2 * (p_new[i, j+1] + p_new[i, j-1]) -
        #                  dx**2 * dy**2 * b_i_j[i, j]) /
        #                 (2 * (dx**2 + dy**2))
        #             )
        #     p_new = BCs.pressure_BC(p_new)
        # return p_new
        
        for _ in range(nppi):
            # p_prev = p_new.copy()
            p_new[1:-1, 1:-1] = (
                    (dy**2 * (p_new[2:, 1:-1] + p_new[:-2, 1:-1]) +
                    dx**2 * (p_new[1:-1, 2:] + p_new[1:-1, :-2]) -
                    dx**2 * dy**2 * b_i_j[1:-1, 1:-1]) /
                    (2 * (dx**2 + dy**2))
            )
            p_new = BCs.pressure_BC(p_new)
            p_prev = p_new
        return p_new

class AdvectionVelocityCorrection:
    @staticmethod
    def velocity_correction_horizontal(ux: np.ndarray, dp__dx: np.ndarray, rho_fluid: float, dt: float) -> np.ndarray:
        """Correct the velocity after pressure calculation."""
        """"the pressure correction for the x-component of velocity based on the pressure gradient in the x-direction."""
        ux_new = (ux - (dt / rho_fluid) * dp__dx)
        return ux_new
        
    @staticmethod
    def velocity_correction_vertical(uy: np.ndarray, dp__dy: np.ndarray, rho_fluid: float, dt: float, 
                           buoyancy:bool=True, beta:float=None, T_next:np.ndarray=None, T_ref:np.ndarray=None, g=9.81) -> np.ndarray:
        """Correct the velocity after pressure calculation."""
        # if buoyancy: # means if buoyancy = True
        #     uy_new = (uy - (dt / rho_fluid) * dp__dy) + beta * (T_next) * dt
        #     return uy_new
        # elif buoyancy==False:
        #     uy_new = (uy - (dt / rho_fluid) * dp__dy)
        #     return uy_new
        
        # in the below code, The initial correction uy - (dt / rho_fluid) * dp__dy
        # is applied regardless of whether buoyancy is considered.
        
        uy_new = uy - (dt / rho_fluid) * dp__dy
        
        if buoyancy:
            if T_next is None or beta is None or T_ref is None:
                raise ValueError("T, beta, and T_ref must be provided when buoyancy is True")
            
            # Calculate density variation
            rho_fluid_new = rho_fluid * (1 - beta * (T_next - T_ref))
            
            # Calculate buoyancy force
            buoyancy_force = g * (rho_fluid - rho_fluid_new) / rho_fluid
            
            # Add buoyancy effect to vertical velocity
            uy_new += buoyancy_force * dt
        
        return uy_new

class PressurePoisson:
    @staticmethod
    def laplace_pressure_2d(p_prev: np.ndarray, dx: float, dy: float, nx:float, ny:float, nppi:float):
        y = np.linspace(0, 1, ny)
        for _ in tqdm (range(nppi)):
            p_new = np.zeros_like(p_prev)
            p_new[1:-1, 1:-1] = ((dy**2 * (p_prev[1:-1, 2:] + p_prev[1:-1, 0:-2]) +
                         dx**2 * (p_prev[2:, 1:-1] + p_prev[0:-2, 1:-1])) /
                        (2 * (dx**2 + dy**2)))
            
            p_new[-1,:] = p_new[-2,0] # Top boundary
            p_new[0,:] = p_new[1,:]# bottom boundary condition
            p_new[:,0] = 0 # left boundary condition, dp/dy = 0 @ y = 0
            p_new[:,-1] = y # right boundary condition, dp/dy = 0 @ y = 1
            
            # Advances in time
            p_prev = p_new
            
        return p_new
    
    @staticmethod
    def laplace_poisson_pressure_2d(p_prev: np.ndarray, b_i_j: np.ndarray, dx: float, dy: float, nx:float, ny:float, nppi:float):
        p_new = np.zeros_like(p_prev)
        for _ in tqdm (range(nppi)):
            p_prev = p_new.copy()
            p_new[1:-1, 1:-1] = (((p_prev[1:-1, 2:] + p_prev[1:-1, :-2]) * dy**2 +
                                (p_prev[2:, 1:-1] + p_prev[:-2, 1:-1]) * dx**2 -
                                b_i_j[1:-1, 1:-1] * dx**2 * dy**2) / 
                                (2 * (dx**2 + dy**2)))
            
            p_new[ny-1,:] = 0 # Top boundary
            p_new[0,:] = 0 # bottom boundary condition
            p_new[:,0] = 0 # left boundary condition, dp/dy = 0 @ y = 0
            p_new[:,nx-1] = 0 # right boundary condition, dp/dy = 0 @ y = 1
            
            # # Advances in time
            # p_prev = p_new
            
        return p_new
    
    @staticmethod
    def source_laplace_poisson_pressure_time_2d(b_i_j: np.ndarray, ux_prev:np.ndarray, uy_prev:np.ndarray, 
                                                rho_fluid: float, dt:float, dx: float, dy: float):
        b_i_j[1:-1, 1:-1] = (rho_fluid * (1 / dt * 
                    ((ux_prev[1:-1, 2:] - ux_prev[1:-1, 0:-2]) / 
                     (2 * dx) + (uy_prev[2:, 1:-1] - uy_prev[0:-2, 1:-1]) / (2 * dy)) -
                    ((ux_prev[1:-1, 2:] - ux_prev[1:-1, 0:-2]) / (2 * dx))**2 -
                      2 * ((ux_prev[2:, 1:-1] - ux_prev[0:-2, 1:-1]) / (2 * dy) *
                           (uy_prev[1:-1, 2:] - uy_prev[1:-1, 0:-2]) / (2 * dx))-
                          ((uy_prev[2:, 1:-1] - uy_prev[0:-2, 1:-1]) / (2 * dy))**2))

        return b_i_j
    
    @staticmethod
    def source_laplace_poisson_pressure_time_periodic_ux_2d(b_i_j: np.ndarray, ux_prev:np.ndarray, uy_prev:np.ndarray, 
                                                rho_fluid: float, dt:float, dx: float, dy: float):
        b_i_j[1:-1, 1:-1] = (rho_fluid * (1 / dt * 
                    ((ux_prev[1:-1, 2:] - ux_prev[1:-1, 0:-2]) / 
                     (2 * dx) + (uy_prev[2:, 1:-1] - uy_prev[0:-2, 1:-1]) / (2 * dy)) -
                    ((ux_prev[1:-1, 2:] - ux_prev[1:-1, 0:-2]) / (2 * dx))**2 -
                      2 * ((ux_prev[2:, 1:-1] - ux_prev[0:-2, 1:-1]) / (2 * dy) *
                           (uy_prev[1:-1, 2:] - uy_prev[1:-1, 0:-2]) / (2 * dx))-
                          ((uy_prev[2:, 1:-1] - uy_prev[0:-2, 1:-1]) / (2 * dy))**2))
        
        b_i_j[1:-1, 0] = (rho_fluid * (1 / dt * 
                    ((ux_prev[1:-1, 1] - ux_prev[1:-1, -1]) / 
                     (2 * dx) + (uy_prev[2:, 0] - uy_prev[0:-2, 0]) / (2 * dy)) -
                    ((ux_prev[1:-1, 1] - ux_prev[1:-1, -1]) / (2 * dx))**2 -
                      2 * ((ux_prev[2:, 0] - ux_prev[0:-2, 0]) / (2 * dy) *
                           (uy_prev[1:-1, 1] - uy_prev[1:-1, -1]) / (2 * dx))-
                          ((uy_prev[2:, -1] - uy_prev[0:-2, -1]) / (2 * dy))**2))
        
        b_i_j[1:-1, -1] = (rho_fluid * (1 / dt * 
                    ((ux_prev[1:-1, 0] - ux_prev[1:-1, -2]) / 
                     (2 * dx) + (uy_prev[2:, -1] - uy_prev[0:-2, -1]) / (2 * dy)) -
                    ((ux_prev[1:-1, 0] - ux_prev[1:-1, -2]) / (2 * dx))**2 -
                      2 * ((ux_prev[2:, -1] - ux_prev[0:-2, -1]) / (2 * dy) *
                           (uy_prev[1:-1, 0] - uy_prev[1:-1, -2]) / (2 * dx))-
                          ((uy_prev[2:, -1] - uy_prev[0:-2, -1]) / (2 * dy))**2))

        return b_i_j

    @staticmethod
    def laplace_poisson_pressure_advance_2d(p_prev: np.ndarray, b_i_j: np.ndarray, dx: float, dy: float, nx:float, ny:float, nppi:float):
        # for _ in tqdm (range(nppi)):
        for _ in range(nppi):
            p_new = np.zeros_like(p_prev)
            
            p_new[1:-1, 1:-1] = (((p_prev[1:-1, 2:] + p_prev[1:-1, 0:-2]) * dy**2 + 
                                (p_prev[2:, 1:-1] + p_prev[0:-2, 1:-1]) * dx**2) /
                                (2 * (dx**2 + dy**2)) -
                                dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * 
                                b_i_j[1:-1,1:-1])
            
            # p_new[:, -1] = p_new[:, -2] # dp/dx = 0 at x = 2
            # p_new[0, :] = p_new[1, :]   # dp/dy = 0 at y = 0
            # p_new[:, 0] = p_new[:, 1]   # dp/dx = 0 at x = 0
            # p_new[-1, :] = 0        # p = 0 at y = 2
            p_new[-1, :] = p_new[-2, :]  # dp/dy = 0 at y = 2
            p_new[0, :] = p_new[1, :]   # dp/dy = 0 at y = 0
            
            # Advances in time
            p_prev = p_new
            
        return p_new
    
    @staticmethod
    def laplace_poisson_pressure_advance_periodic_ux_2d(p_prev: np.ndarray, b_i_j: np.ndarray, dx: float, dy: float, nx:float, ny:float, nppi:float):
        # for _ in tqdm (range(nppi)):
        for _ in range(nppi):
            p_new = np.zeros_like(p_prev)
            p_prev = p_new.copy()
            p_new[1:-1, 1:-1] = (((p_prev[1:-1, 2:] + p_prev[1:-1, 0:-2]) * dy**2 + 
                                (p_prev[2:, 1:-1] + p_prev[0:-2, 1:-1]) * dx**2) /
                                (2 * (dx**2 + dy**2)) -
                                dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * 
                                b_i_j[1:-1,1:-1])
            
            p_new[1:-1, 0] = (((p_prev[1:-1, 1] + p_prev[1:-1, -1]) * dy**2 + 
                                (p_prev[2:, 0] + p_prev[0:-2, 0]) * dx**2) /
                                (2 * (dx**2 + dy**2)) -
                                dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * 
                                b_i_j[1:-1,0])
            
            p_new[1:-1, -1] = (((p_prev[1:-1, 0] + p_prev[1:-1, -2]) * dy**2 + 
                                (p_prev[2:, -1] + p_prev[0:-2, -1]) * dx**2) /
                                (2 * (dx**2 + dy**2)) -
                                dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * 
                                b_i_j[1:-1,-1])
            
            # p_new[:, -1] = p_new[:, -2] # dp/dx = 0 at x = 2
            # p_new[0, :] = p_new[1, :]   # dp/dy = 0 at y = 0
            # p_new[:, 0] = p_new[:, 1]   # dp/dx = 0 at x = 0
            # p_new[-1, :] = 0        # p = 0 at y = 2
            p_new[-1, :] = p_new[-2, :]  # dp/dy = 0 at y = 2
            p_new[0, :] = p_new[1, :]   # dp/dy = 0 at y = 0
            
            # Advances in time
            # p_prev = p_new
            
        return p_new
    
    @staticmethod
    def laplace_poisson_pressure_horizontal_flow(ux_prev: np.ndarray, ux_new:np.ndarray, uy_prev: np.ndarray, p_new:np.ndarray, dx: float,
                                                 dy:float, dt:float, rho_fluid:float, nu:float, F:float):
        ux_new[1:-1, 1:-1] = (ux_prev[1:-1, 1:-1]-
                         ux_prev[1:-1, 1:-1] * dt / dx *
                        (ux_prev[1:-1, 1:-1] - ux_prev[1:-1, 0:-2]) -
                         uy_prev[1:-1, 1:-1] * dt / dy *
                        (ux_prev[1:-1, 1:-1] - ux_prev[0:-2, 1:-1]) -
                         dt / (2 * rho_fluid * dx) * (p_new[1:-1, 2:] - p_new[1:-1, 0:-2]) +
                         nu * (dt / dx**2 *
                        (ux_prev[1:-1, 2:] - 2 * ux_prev[1:-1, 1:-1] + ux_prev[1:-1, 0:-2]) +
                         dt / dy**2 *
                        (ux_prev[2:, 1:-1] - 2 * ux_prev[1:-1, 1:-1] + ux_prev[0:-2, 1:-1]))) + F * dt
        return ux_new
    
    @staticmethod
    def laplace_poisson_pressure_periodic_horizontal_flow(ux_prev: np.ndarray, ux_new:np.ndarray, uy_prev: np.ndarray, p_new:np.ndarray, dx: float,
                                                 dy:float, dt:float, rho_fluid:float, nu:float, F:float):
        ux_new[1:-1, 1:-1] = (ux_prev[1:-1, 1:-1]-
                         ux_prev[1:-1, 1:-1] * dt / dx *
                        (ux_prev[1:-1, 1:-1] - ux_prev[1:-1, 0:-2]) -
                         uy_prev[1:-1, 1:-1] * dt / dy *
                        (ux_prev[1:-1, 1:-1] - ux_prev[0:-2, 1:-1]) -
                         dt / (2 * rho_fluid * dx) * (p_new[1:-1, 2:] - p_new[1:-1, 0:-2]) +
                         nu * (dt / dx**2 *
                        (ux_prev[1:-1, 2:] - 2 * ux_prev[1:-1, 1:-1] + ux_prev[1:-1, 0:-2]) +
                         dt / dy**2 *
                        (ux_prev[2:, 1:-1] - 2 * ux_prev[1:-1, 1:-1] + ux_prev[0:-2, 1:-1])) + F * dt)
        
        ux_new[1:-1, 0] = (ux_prev[1:-1, 0]-
                         ux_prev[1:-1, 0] * dt / dx *
                        (ux_prev[1:-1, 0] - ux_prev[1:-1, -1]) -
                         uy_prev[1:-1, 0] * dt / dy *
                        (ux_prev[1:-1, 0] - ux_prev[0:-2, 0]) -
                         dt / (2 * rho_fluid * dx) * (p_new[1:-1, 1] - p_new[1:-1, -1]) +
                         nu * (dt / dx**2 *
                        (ux_prev[1:-1, 1] - 2 * ux_prev[1:-1, 0] + ux_prev[1:-1, -1]) +
                         dt / dy**2 *
                        (ux_prev[2:, 0] - 2 * ux_prev[1:-1, 0] + ux_prev[0:-2, 0])) + F * dt)
        
        ux_new[1:-1, -1] = (ux_prev[1:-1, -1]-
                         ux_prev[1:-1, -1] * dt / dx *
                        (ux_prev[1:-1, -1] - ux_prev[1:-1, -2]) -
                         uy_prev[1:-1, -1] * dt / dy *
                        (ux_prev[1:-1, -1] - ux_prev[0:-2, -1]) -
                         dt / (2 * rho_fluid * dx) * (p_new[1:-1, 0] - p_new[1:-1, -2]) +
                         nu * (dt / dx**2 *
                        (ux_prev[1:-1, 0] - 2 * ux_prev[1:-1, -1] + ux_prev[1:-1, -2]) +
                         dt / dy**2 *
                        (ux_prev[2:, -1] - 2 * ux_prev[1:-1, -1] + ux_prev[0:-2, -1])) + F * dt)
        return ux_new

    @staticmethod
    def laplace_poisson_pressure_vertical_flow(ux_prev: np.ndarray, uy_prev: np.ndarray, uy_new: np.ndarray, p_new:np.ndarray, dx: float,
                                               dy:float, dt:float, rho_fluid:float, nu:float):
        uy_new[1:-1,1:-1] = (uy_prev[1:-1, 1:-1] -
                        ux_prev[1:-1, 1:-1] * dt / dx *
                       (uy_prev[1:-1, 1:-1] - uy_prev[1:-1, 0:-2]) -
                        uy_prev[1:-1, 1:-1] * dt / dy *
                       (uy_prev[1:-1, 1:-1] - uy_prev[0:-2, 1:-1]) -
                        dt / (2 * rho_fluid * dy) * (p_new[2:, 1:-1] - p_new[0:-2, 1:-1]) +
                        nu * (dt / dx**2 *
                       (uy_prev[1:-1, 2:] - 2 * uy_prev[1:-1, 1:-1] + uy_prev[1:-1, 0:-2]) +
                        dt / dy**2 *
                       (uy_prev[2:, 1:-1] - 2 * uy_prev[1:-1, 1:-1] + uy_prev[0:-2, 1:-1])))
        return uy_new
    
    @staticmethod
    def laplace_poisson_pressure_periodic_vertical_flow(ux_prev: np.ndarray, uy_prev: np.ndarray, uy_new: np.ndarray, p_new:np.ndarray, dx: float,
                                               dy:float, dt:float, rho_fluid:float, nu:float):
        uy_new[1:-1,1:-1] = (uy_prev[1:-1, 1:-1] -
                        ux_prev[1:-1, 1:-1] * dt / dx *
                       (uy_prev[1:-1, 1:-1] - uy_prev[1:-1, 0:-2]) -
                        uy_prev[1:-1, 1:-1] * dt / dy *
                       (uy_prev[1:-1, 1:-1] - uy_prev[0:-2, 1:-1]) -
                        dt / (2 * rho_fluid * dy) * (p_new[2:, 1:-1] - p_new[0:-2, 1:-1]) +
                        nu * (dt / dx**2 *
                       (uy_prev[1:-1, 2:] - 2 * uy_prev[1:-1, 1:-1] + uy_prev[1:-1, 0:-2]) +
                        dt / dy**2 *
                       (uy_prev[2:, 1:-1] - 2 * uy_prev[1:-1, 1:-1] + uy_prev[0:-2, 1:-1])))
        
        uy_new[1:-1,0] = (uy_prev[1:-1, 0] -
                        ux_prev[1:-1, 0] * dt / dx *
                       (uy_prev[1:-1, 0] - uy_prev[1:-1, -1]) -
                        uy_prev[1:-1, 0] * dt / dy *
                       (uy_prev[1:-1, 0] - uy_prev[0:-2, 0]) -
                        dt / (2 * rho_fluid * dy) * (p_new[2:, 0] - p_new[0:-2, 0]) +
                        nu * (dt / dx**2 *
                       (uy_prev[1:-1, 1] - 2 * uy_prev[1:-1, 0] + uy_prev[1:-1, -1]) +
                        dt / dy**2 *
                       (uy_prev[2:, 0] - 2 * uy_prev[1:-1, 0] + uy_prev[0:-2, 0])))
        
        uy_new[1:-1,-1] = (uy_prev[1:-1, -1] -
                        ux_prev[1:-1, -1] * dt / dx *
                       (uy_prev[1:-1, -1] - uy_prev[1:-1, -2]) -
                        uy_prev[1:-1, -1] * dt / dy *
                       (uy_prev[1:-1, -1] - uy_prev[0:-2, -1]) -
                        dt / (2 * rho_fluid * dy) * (p_new[2:, -1] - p_new[0:-2, -1]) +
                        nu * (dt / dx**2 *
                       (uy_prev[1:-1, 0] - 2 * uy_prev[1:-1, -1] + uy_prev[1:-1, -2]) +
                        dt / dy**2 *
                       (uy_prev[2:, -1] - 2 * uy_prev[1:-1, -1] + uy_prev[0:-2, -1])))
        return uy_new
        
    
class PressurePoissonGaussSeidelSimple:
    @staticmethod
    def source_term_pressure_poisson_gauss_seidel_simple(dux_tent__dx:np.ndarray, duy_tent__dy:np.ndarray, rho_fluid:float, dt:float):
        b_i_j = (rho_fluid / dt *(dux_tent__dx + duy_tent__dy))
        return b_i_j
    
    @staticmethod
    def pressure_poisson_gauss_seidel_simple(p_prev: np.ndarray, dx: float, b_i_j: np.ndarray, nppi: int) -> np.ndarray:
        """Solve the pressure Poisson equation using the Jacobi method."""
        for _ in range(nppi):
            p_new = p_prev.copy()
            p_new[1:-1, 1:-1] = 1/4 * (+p_prev[1:-1, 0:-2] + p_prev[0:-2, 1:-1] +
                                        p_prev[1:-1, 2:] + p_prev[2:, 1:-1] - dx**2 * b_i_j[1:-1, 1:-1])
            # p_new[:, -1] = p_new[:, -2] # dp/dx = 0 at x = 2
            # p_new[0, :] = p_new[1, :]   # dp/dy = 0 at y = 0
            # p_new[:, 0] = p_new[:, 1]   # dp/dx = 0 at x = 0
            # p_new[-1, :] = 0        # p = 0 at y = 2
            p_new = BCs.pressure_BC(p_new)
        return p_new
    
class ConverisonEquations():
    