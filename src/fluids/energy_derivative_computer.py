

from src.fluids.convective_mediums import TwoPhaseHydrogen
from src.fluids.hydrogen_retrievers import TwoPhaseRequester


class EnergyDerivativeComputer:

    requester = TwoPhaseRequester()

    def compute_energy_derivative(
        self, hydrogen: TwoPhaseHydrogen, fill: float
    ) -> float:
        dU_dP = self.compute_dU_dP(hydrogen, fill)
        density = self.compute_density(hydrogen, fill)
        return 1 / (density * dU_dP)

    def compute_density(
        self, hydrogen: TwoPhaseHydrogen, fill: float
    ) -> float:
        quality = self.compute_quality(hydrogen, fill)
        return (    
            quality / hydrogen.gas.density
            + (1 - quality) / hydrogen.liquid.density
        ) ** (-1) 

    def compute_dU_dP(
        self, hydrogen: TwoPhaseHydrogen, fill: float
    ) -> float:
        reference_hydrogen = self.get_reference_hydrogen(hydrogen)
        u1 = self.compute_internal_energy(reference_hydrogen, fill)
        u2 = self.compute_internal_energy(hydrogen, fill)
        p1 = reference_hydrogen.pressure
        p2 = hydrogen.pressure
        return (u1 - u2) / (p1 - p2)

    def get_reference_hydrogen(
        self, hydrogen: TwoPhaseHydrogen, pressure_factor: float=1.01
    ) -> TwoPhaseHydrogen:
        return self.requester.get_hydrogen_properties(
            hydrogen.pressure * pressure_factor, None
        )

    def compute_internal_energy(
        self, hydrogen: TwoPhaseHydrogen, fill: float
    ) -> float:
        quality = self.compute_quality(hydrogen, fill)
        return (
            quality * hydrogen.gas.internal_energy
            + (1 - quality) * hydrogen.liquid.internal_energy
        )

    @staticmethod
    def compute_quality(
        hydrogen: TwoPhaseHydrogen, fill: float
    ) -> float:
        factor1 = hydrogen.liquid.density / hydrogen.gas.density
        factor2 = fill / (1 - fill)
        return (1 + factor1 * factor2) ** -1


def main():
    pass


if __name__ == "__main__":
    main()


# End
