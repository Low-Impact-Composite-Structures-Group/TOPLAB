"""Constituent-based thermal capacitance of an MLI blanket (validation only).

Builds the bulk density and mass-weighted specific heat of a Mylar/Dacron
blanket from per-layer areal masses. Nothing here is imported by production.

Sources
-------
* Hastings, Hedayat, Brown, NASA/TM-2004-213175 (2004): Mylar density
  1390 kg/m^3 (Table 1); Dacron netting separator "density" 12.03 kg/m^3 at an
  average of 12 layers/cm (1.105 m^3, 13.3 kg incl. bumpers); MHTB blanket mass
  breakdown (Table 2): 16.6 kg Mylar, 12 kg netting, 1.36 kg bumpers,
  2.72 kg tape for 32.68 kg total. The report has no heat-capacity data.
* Aluminium specific heat: NIST 6061-T6 polynomial already used in production
  (``NISTMetal.aluminum_6061T6_nist``).

ASSUMPTIONS (not from a verified source)
----------------------------------------
* Aluminium coating 400 Angstrom per film side, two sides (typical DAM value).
* One netting sheet per shield, with areal mass NETTING_AREAL_MASS derived from
  the NASA separator density divided by its average layer density.
* ``PET_CP_ANCHORS``: order-of-magnitude recollection of semicrystalline PET
  specific heat. NO table could be retrieved and verified. It is used only as a
  bracketing sensitivity case, never as a recommended property.
"""

from dataclasses import dataclass

import numpy as np

from toplab.materials.nist_materials import NISTMetal

MYLAR_DENSITY = 1390.0
ALUMINIUM_DENSITY = 2700.0
AL_COATING_THICKNESS = 2 * 400e-10
NASA_SEPARATOR_DENSITY = 12.03
NASA_AVERAGE_LAYERS_PER_M = 1200.0
NETTING_AREAL_MASS = NASA_SEPARATOR_DENSITY / NASA_AVERAGE_LAYERS_PER_M

PET_CP_ANCHORS = (  # (T [K], cp [J/kg/K]) -- UNVERIFIED, see module docstring
    (20.0, 40.0),
    (50.0, 200.0),
    (100.0, 500.0),
    (200.0, 900.0),
    (300.0, 1150.0),
)
_ALUMINIUM = NISTMetal.aluminum_6061T6_nist()


def pet_specific_heat(temperature):
    anchors = np.asarray(PET_CP_ANCHORS)
    return float(
        np.exp(
            np.interp(
                temperature, anchors[:, 0], np.log(anchors[:, 1])
            )
        )
    )


def aluminium_specific_heat(temperature):
    return float(_ALUMINIUM.get_specific_heat(max(4.0, min(temperature, 400.0))))


@dataclass(frozen=True)
class BlanketComposition:
    film_thickness: float  # m of Mylar per shield
    layers_per_m: float

    @property
    def areal_masses(self):
        """kg/m^2 per shield: (PET film, PET netting, aluminium)."""
        return (
            MYLAR_DENSITY * self.film_thickness,
            NETTING_AREAL_MASS,
            ALUMINIUM_DENSITY * AL_COATING_THICKNESS,
        )

    @property
    def bulk_density(self):
        return sum(self.areal_masses) * self.layers_per_m

    def specific_heat(self, temperature):
        film, net, aluminium = self.areal_masses
        pet = pet_specific_heat(temperature)
        return (
            (film + net) * pet + aluminium * aluminium_specific_heat(temperature)
        ) / (film + net + aluminium)

    def node_mass_per_area(self, layers_in_node):
        return layers_in_node * sum(self.areal_masses)


QUARTER_MIL = BlanketComposition(6.35e-6, 1600.0)
HALF_MIL = BlanketComposition(12.7e-6, 1600.0)


class TabulatedSpecificHeat:
    """Fast cp(T) and enthalpy H(T) = integral of cp from 4 K (tabulated)."""

    def __init__(self, composition, t_min=4.0, t_max=400.0, points=1981):
        self.temperatures = np.linspace(t_min, t_max, points)
        self.values = np.array(
            [composition.specific_heat(t) for t in self.temperatures]
        )
        increments = 0.5 * (self.values[1:] + self.values[:-1]) * np.diff(
            self.temperatures
        )
        self.enthalpy_table = np.concatenate(([0.0], np.cumsum(increments)))

    def __call__(self, temperature):
        return float(np.interp(temperature, self.temperatures, self.values))

    def enthalpy(self, temperature):
        return np.interp(temperature, self.temperatures, self.enthalpy_table)
