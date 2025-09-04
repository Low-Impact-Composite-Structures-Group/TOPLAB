"""
Enhanced thermal model with direct multi-step heat transfer calculation.

This model implements the direct approach:
1. Heat flow through insulation: Q_insulation = k_insulation * A * (T_amb - T_structure)  
2. Heat flow through tank wall: Q_wall = k_wall * A * (T_structure - T_hydrogen)

Where:
- k_wall = 0.2 W/(m²·K) (assumption for structure to hydrogen)
- k_insulation comes from the SimplifiedThermodynamicModel
- T_structure is calculated from energy balance
"""

import numpy as np
from typing import Tuple, Optional

class DirectThermalModel:
    """
    Direct thermal model implementing multi-step heat transfer calculation.
    """
    
    def __init__(self, k_insulation: float = 0.033, k_wall: float = 0.2):
        """
        Initialize direct thermal model.
        
        Args:
            k_insulation: Overall insulation heat transfer coefficient [W/(m²·K)]
            k_wall: Wall heat transfer coefficient (structure to hydrogen) [W/(m²·K)]
        """
        self.k_insulation = k_insulation
        self.k_wall = k_wall
        self.T_structure = None  # Will be calculated
        
    def compute_structure_temperature(self, T_amb: float, T_hydrogen: float, 
                                    tank_thermal_capacity: float, dt: float = 1.0) -> float:
        """
        Calculate structure temperature from energy balance.
        
        For steady-state: Q_insulation = Q_wall
        k_insulation * (T_amb - T_structure) = k_wall * (T_structure - T_hydrogen)
        
        Solving for T_structure:
        T_structure = (k_insulation * T_amb + k_wall * T_hydrogen) / (k_insulation + k_wall)
        
        Args:
            T_amb: Ambient temperature [K]
            T_hydrogen: Hydrogen temperature [K] 
            tank_thermal_capacity: Tank thermal capacity [J/K]
            dt: Time step [s]
            
        Returns:
            Structure temperature [K]
        """
        # Steady-state approximation for structure temperature
        numerator = self.k_insulation * T_amb + self.k_wall * T_hydrogen
        denominator = self.k_insulation + self.k_wall
        T_structure = numerator / denominator
        
        self.T_structure = T_structure
        return T_structure
        
    def compute_heat_fluxes(self, tank, tank_state, mission_section) -> Tuple[float, float, float, list]:
        """
        Compute heat fluxes using direct multi-step approach.
        
        Args:
            tank: Tank object with surface area property
            tank_state: Tank state with temperature and thermal capacity
            mission_section: Mission section with ambient temperature
            
        Returns:
            tuple: (q_insulation [W/m²], q_wall [W/m²], q_net [W/m²], temperature_profile [K])
        """
        # Get temperatures
        T_amb = mission_section.temperature  # Ambient temperature [K]
        T_H2 = tank_state.temperature       # Hydrogen temperature [K]
        
        # Get dynamic tank thermal capacity (temperature-dependent)
        tank_thermal_capacity = self._get_dynamic_thermal_capacity(tank, tank_state)
        
        # Calculate structure temperature
        T_structure = self.compute_structure_temperature(T_amb, T_H2, tank_thermal_capacity)
        
        # Calculate heat fluxes [W/m²]
        q_insulation = self.k_insulation * (T_amb - T_structure)
        q_wall = self.k_wall * (T_structure - T_H2)
        
        # Net heat flux to hydrogen
        q_net = q_wall
        
        # Temperature profile: [T_H2, T_structure, T_amb]
        temperature_profile = [T_H2, T_structure, T_amb]
        
        # Debug output
        if abs(T_amb - T_H2) > 1e-6:
            print(f"\n=== DIRECT THERMAL MODEL ===")
            print(f"Ambient temperature: {T_amb:.1f} K")
            print(f"Structure temperature: {T_structure:.1f} K")
            print(f"Hydrogen temperature: {T_H2:.1f} K")
            print(f"k_insulation: {self.k_insulation:.6f} W/(m²·K)")
            print(f"k_wall: {self.k_wall:.3f} W/(m²·K)")
            print(f"Heat flux through insulation: {q_insulation:.3f} W/m²")
            print(f"Heat flux through wall: {q_wall:.3f} W/m²")
            print(f"Net heat flux to hydrogen: {q_net:.3f} W/m²")
            
            # Calculate total heat rates
            if hasattr(tank, 'surface_area'):
                Q_insulation_total = q_insulation * tank.surface_area
                Q_wall_total = q_wall * tank.surface_area
                Q_net_total = q_net * tank.surface_area
                print(f"Total heat rate through insulation: {Q_insulation_total:.2f} W")
                print(f"Total heat rate through wall: {Q_wall_total:.2f} W")
                print(f"Total net heat rate: {Q_net_total:.2f} W")
                
                # Display tank thermal capacity
                print(f"Tank thermal capacity: {tank_thermal_capacity:.1f} J/K")
            print(f"==============================")
        
        return q_insulation, q_wall, q_net, temperature_profile
        
    def _get_dynamic_thermal_capacity(self, tank, tank_state) -> float:
        """
        Get dynamic tank thermal capacity based on current structure temperature.
        
        Args:
            tank: Tank object
            tank_state: Current tank state
            
        Returns:
            Tank thermal capacity [J/K]
        """
        # Use structure temperature if available, otherwise use hydrogen temperature as approximation
        structure_temp = self.T_structure if self.T_structure is not None else tank_state.temperature
        
        # Get temperature-dependent thermal capacity from NIST data
        try:
            thermal_capacity = tank.compute_thermal_capacity(structure_temp)
            return thermal_capacity
        except Exception as e:
            # Fallback to constant value if NIST calculation fails
            print(f"Warning: Could not compute dynamic thermal capacity, using fallback: {e}")
            return 1000.0  # J/K - reasonable fallback value
            
    def compute_heat_flux(self, tank, tank_state, mission_section) -> Tuple[float, list]:
        """
        Compatibility method for SimplifiedThermodynamicModel interface.
        
        Returns:
            tuple: (net_heat_flux [W/m²], temperature_profile [list])
        """
        q_insulation, q_wall, q_net, temperature_profile = self.compute_heat_fluxes(
            tank, tank_state, mission_section
        )
        
        return q_net, temperature_profile


class EnhancedSimplifiedThermodynamicModel(DirectThermalModel):
    """
    Enhanced version of SimplifiedThermodynamicModel with direct thermal approach.
    """
    
    def __init__(self, k_insulation: float = 0.033, k_wall: float = 0.2):
        """
        Initialize enhanced simplified model.
        
        Args:
            k_insulation: Overall insulation heat transfer coefficient [W/(m²·K)]
            k_wall: Wall heat transfer coefficient [W/(m²·K)]
        """
        super().__init__(k_insulation, k_wall)
        
        # Compatibility attributes for existing code
        self.k_amb = k_insulation  # For backward compatibility
        self.insulation_layers = 1
        self.liner_layers = 1
        self.max_iterations = 1
        self.constant_heat_flux = None
        
    def compute_thermal_resistances(self, tank, tank_state, mission_section, temperatures):
        """Compatibility method - not used in direct approach."""
        return []
        
    def _compute_total_temperature_interfaces(self, tank):
        """Compatibility method - direct model uses 3 temperature points."""
        return 3
        
    def define_initial_temperatures(self, tank_temp, ambient_temp, num_interfaces):
        """Define temperature profile for compatibility."""
        # Calculate initial structure temperature estimate
        T_structure_init = (tank_temp + ambient_temp) / 2
        return [tank_temp, T_structure_init, ambient_temp]


# Factory function for easy integration
def create_enhanced_thermal_model(k_insulation: float = 0.033, k_wall: float = 0.2):
    """
    Create enhanced thermal model with direct heat transfer approach.
    
    Args:
        k_insulation: Overall insulation heat transfer coefficient [W/(m²·K)]
        k_wall: Wall heat transfer coefficient [W/(m²·K)]
        
    Returns:
        EnhancedSimplifiedThermodynamicModel instance
    """
    print(f"🔥 Creating ENHANCED thermal model")
    print(f"   Insulation coefficient: {k_insulation:.6f} W/(m²·K)")
    print(f"   Wall coefficient: {k_wall:.3f} W/(m²·K)")
    print(f"   Using direct multi-step approach")
    
    return EnhancedSimplifiedThermodynamicModel(k_insulation, k_wall)
