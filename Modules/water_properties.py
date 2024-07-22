import numpy as np

class FluidProperties:
    
    @staticmethod
    def density_pressure(pressure: np.ndarray, temperature: float) -> np.ndarray:
        """
        Calculate fluid density based on pressure and temperature.

        Args:
            pressure (np.ndarray): Pressure values in Pa.
            temperature (float): Temperature in °C.

        Returns:
            np.ndarray: Fluid density in kg/m³.
        """
        density = (pressure / 1490.9) - 0.0026 * (temperature**2) - 0.15594 * temperature + 1001.5515
        return density
    
    @staticmethod
    def viscosity(temperature: float, temp_over_100: bool = True) -> float:
        """
        Calculate fluid viscosity based on temperature.

        Args:
            temperature (float): Temperature in °C.
            temp_over_100 (bool, optional): Whether temperature is over 100°C. Defaults to True.

        Returns:
            float: Fluid viscosity in Pa·s.
        """
        if temp_over_100:
            viscosity = (3.27*(10**-11)*(temperature**4) - 9.14*(10**-9)*(temperature**3) 
                         + 9.93*(10**-7)*(temperature**2) - 5.56*(10**-5)*(temperature) + 1.79*(10**-3))
        else:
            viscosity = (-2.025*(10**-11)*(temperature**3) + 1.72*(10**-8)*(temperature**2) 
                         - 5.18*(10**-6)*(temperature) + 6.44 *(10**-4))
        return viscosity
    
    @staticmethod
    def specific_heat_capacity_p(temperature: float) -> float:
        """
        Calculate specific heat capacity at constant pressure.

        Args:
            temperature (float): Temperature in °C.

        Returns:
            float: Specific heat capacity at constant pressure in J/(kg·K).
        """
        cp = 1.16*(10**-4)*(temperature**3) - 2.35*(10**-2)*(temperature**2) + 1.61*temperature + 4.17*(10**3)
        return cp
    
    @staticmethod
    def specific_heat_capacity_v(temperature: float) -> float:
        """
        Calculate specific heat capacity at constant volume.

        Args:
            temperature (float): Temperature in °C.

        Returns:
            float: Specific heat capacity at constant volume in J/(kg·K).
        """
        cv = 3.4*(10**-5)*(temperature**3) - 1.01*(10**-2)*(temperature**2) + 3.95*temperature + 4.24*(10**3)
        return cv
    
    @staticmethod
    def thermal_conductivity(temperature: float) -> float:
        """
        Calculate thermal conductivity of the fluid.

        Args:
            temperature (float): Temperature in °C.

        Returns:
            float: Thermal conductivity in W/(m·K).
        """
        k = -5.71*(10**-6)*(temperature**2) + 1.64*(10**-3)*temperature + 5.69*(10**-1)
        return k