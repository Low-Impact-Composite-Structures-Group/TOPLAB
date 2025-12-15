"""
Safe wrappers around CoolProp for hydrogen to reduce warnings and handle
near-saturation and two-phase conditions gracefully.

These helpers centralize the common patterns we need:
- Pressure from (T, rho) with two-phase detection → use Psat(T)
- Enthalpy selection that uses saturation relationships in two-phase

Note: Keep these functions lightweight and dependency-free beyond CoolProp.
"""

from __future__ import annotations

from typing import Optional


def safe_pressure_from_T_rho(T: float, rho: float, fluid: str = "hydrogen") -> float:
    """
    Compute pressure robustly from temperature [K] and density [kg/m^3].

    - If T <= 0 or rho <= 0: return 1 bar fallback.
    - If T < Tcrit and rho is between saturated vapor and liquid densities at T,
      return saturation pressure Psat(T) to avoid two-phase inversion warnings.
    - Otherwise use CoolProp PropsSI("P", "T", T, "Dmass", rho, fluid).
    """
    try:
        if T is None or rho is None:
            return 1.0e5
        if T <= 0 or rho <= 0:
            return 1.0e5

        from CoolProp.CoolProp import PropsSI

        # Critical temperature check for two-phase detection
        try:
            Tcrit = PropsSI("Tcrit", fluid)
        except Exception:
            Tcrit = 33.0  # Hydrogen ~33 K (fallback)

        if T < Tcrit - 1e-6:
            # Get saturation densities at this temperature
            try:
                rho_l = PropsSI("Dmass", "T", T, "Q", 0, fluid)
                rho_g = PropsSI("Dmass", "T", T, "Q", 1, fluid)
                # If state lies in the two-phase dome, use saturation pressure
                if rho_g + 1e-9 <= rho <= rho_l - 1e-9:
                    return PropsSI("P", "T", T, "Q", 0, fluid)
            except Exception:
                # Fall back to direct computation below
                pass

        # Default: single-phase computation
        return float(PropsSI("P", "T", T, "Dmass", rho, fluid))

    except Exception:
        # Last resort fallback
        return 1.0e5


def safe_enthalpy(
    T: float,
    P: Optional[float] = None,
    *,
    assume_gas_when_twophase: bool = True,
    fluid: str = "hydrogen",
) -> float:
    """
    Compute mass-specific enthalpy [J/kg] robustly.

    - For two-phase conditions when only T is known, use saturated vapor enthalpy h_g(T)
      if assume_gas_when_twophase is True, else saturated liquid enthalpy h_l(T).
    - If P is provided and T/P pair is valid single-phase, use (T, P) call.
    """
    try:
        if T is None or T <= 0:
            return 0.0

        from CoolProp.CoolProp import PropsSI

        if P is not None and P > 0:
            try:
                return float(PropsSI("Hmass", "T", T, "P", P, fluid))
            except Exception:
                # fall through to two-phase handling below
                pass

        # Two-phase aware fallback using saturation at T
        try:
            if assume_gas_when_twophase:
                return float(PropsSI("Hmass", "T", T, "Q", 1, fluid))
            else:
                return float(PropsSI("Hmass", "T", T, "Q", 0, fluid))
        except Exception:
            # Last resort: idealized estimate using cp*T (very rough)
            try:
                cp = PropsSI("Cpmass", "T", max(T, 1.0), "P", max(P or 1.0e5, 1.0), fluid)
                return float(cp * T)
            except Exception:
                return 0.0

    except Exception:
        return 0.0
