from dataclasses import dataclass, field
import numpy as np
from typing import Optional, Dict, Any
from test import *

@dataclass # provide a way to create classes that are primarily used to store data
class SetupParameters:
    nx: int = 41 # Number of spatial points in x direction
    ny: int = 41 # Number of spatial points in y direction
    lx: float = 2 #  Length of the domain in x direction
    ly: float = 2 #  Length of the domain in y direction
    nt: int = 5000 # Number of time steps
    nppi: int = 50 #N_PRESSURE_POISSON_ITERATIONS
    # nppi denotes the iteration index in the Jacobi method.The jacobi method will iterate 
    # until the solution converges to a desired tolerance 
    tol:int = 1e-5 # tolerance
    
    nu: float = 0.1 # Kinematic viscosity (Pa.s)
    rho_fluid: float = 1 # Density of fluid (at ambient temperature - (kg/ m^3))
    rho_solid: float = 1 # Densioty of solid (kg/ m^3)
    beta: float = 0 # Volumetric thermal expansion coefficient (1/k)
    alpha: float = 1 # Thermal Diffusivity
    c: float = 1 # Wave propagation speed
    reynolds_number: float = 10000 # 
    prandtl_number: float = 0.7 # 
    sf: float = 0.5 # STABILITY_SAFETY_FACTOR
    F: float = 1 # Source value for the channel flow equation
    g:float = 9.81 # gravitational acceleration
    lambda_s: float = 1 # thermal conductivity of solids
    lambda_f: float = 1 # thermal conductivity of fluid
    n: float = 1 # porosity
    k:float = 1 # permeability (m^2)
    c_p_f: float = 1 # specific heat capacity of fluid (J/kg c)
    alpha_fluid:float = 1.39e-7 # Thermal diffusivity of fluid (m^2/s)
    alpha_solid:float = 1 # Thermal diffusivity of soil ()
    
    # Geothermal Properties
    downhole_temperature: float = 0 # C
    thermal_gradient: float = 0 # Thermal Gradient C/m
    heat_capacity_solid:float = 1 # J/kg.K
    heat_capacity_fluid:float = 1 # J/kg.K
    thermal_conductivity_fluid:float = 1 # W/ m.k
    thermal_conductivity_solid:float = 1 # W/ m.k
    
    # Source Properties
    source_radius:float = 1
    source_temp:float = 1
    source_ux:float = 0
    source_uy:float = 0
    
    dt: Optional[float] = None
    dx:Optional[int] = None 
    dy:Optional[int] = None 
    alpha:Optional[float] = None 
    gamma:Optional[float] = None 
    sigma:Optional[float] = None 
    lambda_b:Optional[float] = None
    
    def __post_init__(self): # this allows you to perform additional initialization after the init method initialization 

        if self.dx is None:
            self.dx = self.lx / (self.nx - 1)
        if self.dy is None:
            self.dy = self.ly / (self.ny - 1)
        if self.alpha is None:
            self.alpha = np.sqrt(self.prandtl_number / self.reynolds_number)
        if self.gamma is None:
            self.gamma = 1 / (np.sqrt(self.reynolds_number * self.prandtl_number))
        if self.sigma is None:
            self.sigma = np.sqrt(self.nu)
        if self.dt is None:
            self.dt = (self.sigma * self.dx * self.dy) / self.nu
        if self.lambda_b is None:
            self.lambda_b = self.lambda_s * (1-self.n) + self.lambda_f * self.n
        
    @classmethod
    def from_dict(cls, param_dict: Dict[str, Any]) -> 'SimulationParameters':
        # Filter the input dictionary to only include valid attributes
        valid_params = {k: v for k, v in param_dict.items() if k in cls.__annotations__}
        
        # Create and return a new instance of the class
        return cls(**valid_params)
        # This line creates and returns a new instance of the class.
        # cls(**valid_params) is equivalent to calling the constructor with the filtered parameters as keyword arguments.
        
