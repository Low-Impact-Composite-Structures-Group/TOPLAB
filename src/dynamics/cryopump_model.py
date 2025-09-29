"""
CryoPump model for hydrogen refueling simulation.

This module provides a simple thermodynamic model for calculating the outlet
state of hydrogen after being compressed by a cryogenic pump during refueling.
"""

from CoolProp.CoolProp import PropsSI
from src.fluids.hydrogen_retrievers import SinglePhaseRequester, TwoPhaseRequester


class CryoPumpModel:
    """
    Model for cryogenic pump used in hydrogen refueling.

    This model simulates the thermodynamic process of compressing liquid hydrogen
    from a low-pressure reservoir to the target tank pressure, accounting for
    pump efficiency and resulting temperature rise.
    """

    @classmethod
    def compute_pump_outlet_hydrogen(cls, tank_pressure: float, tank_temperature: float,
                                   dewar_pressure: float = 3e5, pump_efficiency: float = 0.78):
        """
        Calculate the hydrogen properties and pressure at pump outlet.

        Args:
            tank_pressure: Target tank pressure [Pa]
            tank_temperature: Current tank temperature [K]
            dewar_pressure: Source dewar pressure [Pa] (default 3e5 Pa = 3 bar)
            pump_efficiency: Pump isentropic efficiency (default 0.78 = 78%)
        """
        fluid = "Hydrogen"
        P1 = dewar_pressure    # Pa - configurable dewar pressure
        P2 = tank_pressure     # Target pressure (Pa)
        eta_p = pump_efficiency # Configurable pump isentropic efficiency

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

        # Return the enthalpy at the pump outlet
        return h2
