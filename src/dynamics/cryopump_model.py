"""
CryoPump model for hydrogen refueling simulation.

This module provides a simple thermodynamic model for calculating the outlet
state of hydrogen after being compressed by a cryogenic pump during refueling.
"""

from CoolProp.CoolProp import PropsSI
from src.fluids.hydrogen_retrievers import SinglePhaseRequester


class CryoPumpModel:
    """
    Model for cryogenic pump used in hydrogen refueling.

    This model simulates the thermodynamic process of compressing liquid hydrogen
    from a low-pressure reservoir to the target tank pressure, accounting for
    pump efficiency and resulting temperature rise.
    """

    @classmethod
    def compute_pump_outlet_hydrogen(cls, tank_pressure: float, tank_temperature: float):
        """
        Calculate the hydrogen properties and pressure at pump outlet.

        This method simulates a cryogenic pump compressing hydrogen from
        a low-pressure reservoir to the target tank pressure. It accounts for:
        1. Pump isentropic efficiency
        2. Enthalpy rise during compression
        3. Resulting temperature change

        Args:
            tank_pressure: Current pressure in the receiving tank (Pa)
            tank_temperature: Current temperature in the receiving tank (K)

        Returns:
            tuple: (hydrogen_properties, new_pressure)
                - hydrogen_properties: Hydrogen object with properties at pump outlet
                - new_pressure: The outlet pressure (Pa)
        """
        fluid = "Hydrogen"
        P1 = 3e5       # Pa (2 bar) - reservoir pressure
        P2 = tank_pressure  # Target pressure (Pa)
        eta_p = 0.78   # Pump isentropic efficiency (78%)

        # 1. Inlet state: saturated liquid at P1
        h1 = PropsSI("H", "P", P1, "Q", 0, fluid)  # Enthalpy (J/kg)
        s1 = PropsSI("S", "P", P1, "Q", 0, fluid)  # Entropy (J/kg/K)

        # 2. Ideal isentropic outlet at P2
        h2s = PropsSI("H", "P", P2, "S", s1, fluid)

        # 3. Actual outlet enthalpy with efficiency
        h2 = h1 + (h2s - h1)/eta_p

        # 4. Outlet temperature from (h2,P2)
        T2 = PropsSI("T", "P", P2, "H", h2, fluid)

        # Calculate critical point for reference
        p_crit = PropsSI("Pcrit", "", 0, "", 0, "hydrogen")
        T_crit = PropsSI("Tcrit", "", 0, "", 0, "hydrogen")

        # If P2 less than critical point, boost to critical point
        if P2 < p_crit:
            P2 = p_crit
            T2 = T_crit

        # Use SinglePhaseRequester to avoid two-phase detection issues
        hydrogen = SinglePhaseRequester().get_hydrogen_properties(P2, T2)

        # Return both the hydrogen properties and the new pressure to use for the tank
        return hydrogen, P2
