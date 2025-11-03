from .foam_insulations import ConstantFoamInsulation, VariableFoamInsulation, VariableFoamInsulationLinearInterpolated

class InsulationFactory:

    _insulations = {
        "foam_constant": ConstantFoamInsulation,
        "foam_variable": VariableFoamInsulation,
        "foam_variable_linearised": VariableFoamInsulationLinearInterpolated,
    }

    @property
    def _available(self):
        return ", ".join(self._insulations.keys())

    def create_insulation(self, type: str, args: dict):
        if type is None:
            return None
        
        initialiser = self._insulations.get(type)

        if initialiser is None:
            raise ValueError(
                f"'{type}' is an invalid insulation type.\n"
                f"Available types are: {self._available}"
            )
        
        return initialiser(**args)