from __future__ import annotations

from typing import Optional


def safe_pressure_from_T_rho(T: float, rho: float, fluid: str = "hydrogen") -> float:
    try:
        if T is None or rho is None:
            return 1.0e5
        if T <= 0 or rho <= 0:
            return 1.0e5

        from CoolProp.CoolProp import PropsSI

        try:
            critical_temperature = PropsSI("Tcrit", fluid)
        except Exception:
            critical_temperature = 33.0

        if T < critical_temperature - 1e-6:
            try:
                rho_l = PropsSI("Dmass", "T", T, "Q", 0, fluid)
                rho_g = PropsSI("Dmass", "T", T, "Q", 1, fluid)
                if rho_g + 1e-9 <= rho <= rho_l - 1e-9:
                    return PropsSI("P", "T", T, "Q", 0, fluid)
            except Exception:
                pass

        return float(PropsSI("P", "T", T, "Dmass", rho, fluid))
    except Exception:
        return 1.0e5


def safe_enthalpy(
    T: float,
    P: Optional[float] = None,
    *,
    assume_gas_when_twophase: bool = True,
    fluid: str = "hydrogen",
) -> float:
    try:
        if T is None or T <= 0:
            return 0.0

        from CoolProp.CoolProp import PropsSI

        if P is not None and P > 0:
            try:
                return float(PropsSI("Hmass", "T", T, "P", P, fluid))
            except Exception:
                pass

        try:
            quality = 1 if assume_gas_when_twophase else 0
            return float(PropsSI("Hmass", "T", T, "Q", quality, fluid))
        except Exception:
            try:
                cp = PropsSI("Cpmass", "T", max(T, 1.0), "P", max(P or 1.0e5, 1.0), fluid)
                return float(cp * T)
            except Exception:
                return 0.0
    except Exception:
        return 0.0