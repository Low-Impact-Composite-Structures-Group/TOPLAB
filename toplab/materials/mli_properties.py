"""
MLI surrogate: lumped equivalent insulation model.

The bulk equivalent conductivity is fitted as

    k_eq(T) = A*T + B*T**3

so the conductivity integral over [T_cold, T_hot] is

    F(T_hot) - F(T_cold),   F(T) = A*T**2/2 + B*T**4/4.

The geometry factor G is divided out in the fit, so the integral is a bulk
value [W/m] used with the same geometry factors as any other insulation.
A and B have opposite signs and the two terms nearly cancel, so the result is
very sensitive to the coefficients; they must be the exact fitted values.
k_eq changes sign at T = sqrt(-A/B) ~ 208 K (negative below), so the model is
only meaningful as an integral over a range like the fitted one (~25-300 K).
The fit reference is T_hot = outer (shell) surface and T_cold = inner
(structure) surface of the insulation layer.

Mass is lumped: m_ins = EFFECTIVE_DENSITY * insulation volume. This is an
equivalent density, not the density of any real MLI layer; a relation for
m_ins as a function of volume is still to be derived (linear for now).
"""

# Defaults, user-overridable through the YAML insulation block.
DEFAULT_A = -1.134e-5           # [W/m/K^2] = 2 * A* of the fit (A* = -5.67e-6)
DEFAULT_B = 2.616e-10           # [W/m/K^4] = 4 * B* of the fit (B* = +6.54e-11)
DEFAULT_EFFECTIVE_DENSITY = 300.0   # [kg/m^3], fitted equivalent density
DEFAULT_SPECIFIC_HEAT = 1000.0  # [J/kg/K], provisional constant (order of Mylar/Dacron)


def integrated_thermal_conductivity(
    temperature_low: float,
    temperature_high: float,
    A: float = DEFAULT_A,
    B: float = DEFAULT_B,
) -> float:
    """Integrate the MLI equivalent conductivity [W/m] over a temperature interval."""
    def antiderivative(T: float) -> float:
        return A * T ** 2 / 2.0 + B * T ** 4 / 4.0

    return antiderivative(temperature_high) - antiderivative(temperature_low)
