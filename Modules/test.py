# The TestStability module provides a set of static methods for testing and ensuring 
# the stability of the numerical simulation.

from abc import abstractmethod
import numpy as np

class TestStability:
    
    @staticmethod
    def test_initial_timestep_value(dx, dt, nu, sf):
        maximum_possible_time_step_length = (0.5 * dx**2) / nu # von Neumann stability analysis for the diffusion equation
        while dt > sf * maximum_possible_time_step_length:
            print("Stability is not guaranteed: modifying timestep value")
            dt = dt/2
        return dt
    
    # # This function checks and adjusts the time step based on the Courant–Friedrichs–Lewy (CFL) condition
    # @staticmethod
    # def test_CFL_number_calculation(ux_next, uy_next, dx, dy, dt, sf):
    #     CFL_X = np.max(np.abs(ux_next)) * (dt / dx) 
    #     CFL_Y = np.max(np.abs(uy_next)) * (dt / dy)
    #     CFL = max(CFL_X,CFL_Y)
        
    #     if CFL >= 1:
    #         print("Stability not guaranteed: modifying timestep value")
    #         # Adjust dt to satisfy the CFL condition
    #         dt = min(sf * dx / np.max(np.abs(ux_next)), sf * dy / np.max(np.abs(uy_next)))
        
    #     return dt
    
    # in this CFL condition check, if you want to adjust dt in all cases, change the code to following:
    @staticmethod
    def test_CFL_number_calculation(ux_next, uy_next, dx, dy, dt, safety_factor=0.9):
        CFL_X = np.max(np.abs(ux_next)) * (dt / dx) 
        CFL_Y = np.max(np.abs(uy_next)) * (dt / dy)
        CFL = max(CFL_X, CFL_Y)
        
        if CFL >= 1:
            print("Stability not guaranteed: modifying timestep value")
            # Adjust dt to satisfy the CFL condition
            dt_new = safety_factor * min(dx / np.max(np.abs(ux_next)), dy / np.max(np.abs(uy_next)))
        else:
            dt_new = dt
        
        return dt_new, CFL
    
    @staticmethod
    def test_pressure_poisson_convergence(p_prev, p_next, tolerance=1e-5):
        difference = np.mean(np.abs(p_next - p_prev))
        if difference < tolerance:
            return True
        elif difference > 1e3:
            print("Warning: Pressure solution may be diverging")
            return False
        return False
    
    @staticmethod
    def check_peclet_number(ux, uy, dx, dy, nu):
        Pe_x = np.max(np.abs(ux)) * dx / nu
        Pe_y = np.max(np.abs(uy)) * dy / nu
        Pe = max(Pe_x, Pe_y)
        if Pe > 2:
            print(f"Warning: Peclet number ({Pe}) is greater than 2. Numerical diffusion may occur.")
        return Pe
