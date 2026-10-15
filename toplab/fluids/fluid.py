"""Fluid identity: maps a tank's declared fluid label to a CoolProp fluid name."""

from __future__ import annotations

from typing import Optional

DEFAULT_COOLPROP_FLUID = "PARAHYD"

# Storage-state labels used in tank specs. All are hydrogen stored in different
# states, so they share one equation of state.
_LABEL_TO_COOLPROP = {
    "CH2": "PARAHYD",
    "CCH2": "PARAHYD",
    "SLH2": "PARAHYD",
    "LH2": "PARAHYD",
    "H2": "PARAHYD",
}


def resolve_coolprop_fluid(label: Optional[str], override: Optional[str] = None) -> str:
    """
    Return the CoolProp fluid name for a tank.

    ``override`` (a tank's explicit ``coolprop_fluid``) wins; otherwise the label
    must be a known storage label. Unknown labels raise instead of silently
    falling back to hydrogen.
    """
    if override:
        return override
    key = (label or "").upper()
    if key not in _LABEL_TO_COOLPROP:
        raise ValueError(
            f"Unknown tank fluid {label!r}; expected one of {sorted(_LABEL_TO_COOLPROP)} "
            f"or an explicit 'coolprop_fluid'"
        )
    return _LABEL_TO_COOLPROP[key]
