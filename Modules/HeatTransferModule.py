import math

class HeatTransfer:
    @staticmethod
    def convection_coefficient(nusselt_number: float, thermal_conductivity_fluid: float, well_diameter: float) -> float:
        """
        Calculate the convection coefficient.

        Args:
            nusselt_number (float): The Nusselt number (dimensionless).
            thermal_conductivity_fluid (float): The thermal conductivity of the fluid (W/(m·K)).
            well_diameter (float): The diameter of the well (m).

        Returns:
            float: The convection coefficient (W/(m²·K)).
        """
        h = (nusselt_number * thermal_conductivity_fluid) / well_diameter
        return h
    
    @staticmethod
    def resistance_conduction(r0: float, r1: float, thermal_conductivity_solid: float) -> float:
        """
        Calculate the conduction resistance.

        Args:
            r0 (float): Outer radius (m).
            r1 (float): Inner radius (m).
            thermal_conductivity_solid (float): Thermal conductivity of the solid material (W/(m·K)).

        Returns:
            float: The conduction resistance (K/W).
        """
        R_conduction = math.log(r0/r1) / (2 * math.pi * thermal_conductivity_solid)
        return R_conduction
    
    @staticmethod
    def resistance_convection(well_diameter: float, convection_coefficient: float) -> float:
        """
        Calculate the convection resistance.

        Args:
            well_diameter (float): The diameter of the well (m).
            convection_coefficient (float): The convection coefficient (W/(m²·K)).

        Returns:
            float: The convection resistance (K/W).
        """
        R_convection = 1 / (convection_coefficient * math.pi * well_diameter)
        return R_convection