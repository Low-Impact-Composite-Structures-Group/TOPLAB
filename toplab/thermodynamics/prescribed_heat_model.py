from toplab.thermodynamics.isochoric_thermal_model import IsochoricThermalModel


class PrescribedHeatThermalModel(IsochoricThermalModel):
    """
    Replaces the insulation network with a fixed heat input.

    Q_structure_to_h2 = Q_dot (constant) into the hydrogen control volume.
    Structure, insulation, and shell temperatures are pinned.

    Use for D1 verification to isolate hydrogen thermodynamics from
    the external thermal network.
    """

    def __init__(self, Q_dot: float):
        self.Q_dot = Q_dot  # [W], positive = heat into hydrogen

    def compute_structure_to_h2_heat_flux(self, time: float, state, **kwargs) -> float:
        return self.Q_dot

    def compute_structure_temperature_derivative(self, time: float, state, **kwargs) -> float:
        return 0.0

    def compute_insulation_temperature_derivative(self, time: float, state, **kwargs) -> float:
        return 0.0

    def compute_shell_temperature_derivative(self, time: float, state, **kwargs) -> float:
        return 0.0
