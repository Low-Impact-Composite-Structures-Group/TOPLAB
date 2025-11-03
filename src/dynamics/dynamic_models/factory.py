
from . import ahluwalia
from . import lin

class FormulationModelSelector:

    _formulations = {
        "ahluwalia": ahluwalia.DynamicModelFactory(),
        "lin": lin.DynamicModelFactory(),
        "always_two_phase": ahluwalia.AlwaysTwoPhaseFactory(),
    }

    @property
    def _available(self):
        return ", ".join(self._formulations.keys())

    def get_dynamic_model(self, type: str):
        formulation = self._formulations.get(type)

        if formulation is not None:
            return formulation
        
        raise ValueError(
            f"'{type}' is not a valid dynamic model formulation.\n"
            f"Available formulations are: {self._available}"
        )