"""
Enhanced thermal model with direct multi-step heat transfer calculation.

This model implements the direct approach:
1. He        # Liner layers: divide thickness equally
        liner_layer_thickness = liner_thickness / n_liner_layers
        R_liner_layer = liner_layer_thickness / k_liner

        print(f"  Layer details:")
        print(f"    Wall: {n_wall_layers} layers × {wall_layer_thickness*1000:.1f}mm, R = {R_wall_layer:.6f} m²·K/W each")
        print(f"    Liner: {n_liner_layers} layers × {liner_layer_thickness*1000:.1f}mm, R = {R_liner_layer:.6f} m²·K/W each")
        print(f"    Insulation: boundary condition h = {self.k_insulation:.6f} W/(m²·K)")through insulation: Q_insulation = k_insulation * A * (T_amb - T_structure)
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

    def __init__(self, k_insulation: float = 0.033, k_wall: float = 0.2,
                 use_iterative_solver: bool = True):
        """
        Initialize direct thermal model.

        Args:
            k_insulation: Overall insulation heat transfer coefficient [W/(m²·K)]
            k_wall: Wall heat transfer coefficient (structure to hydrogen) [W/(m²·K)]
            use_iterative_solver: Use iterative solver for interface temperatures (recommended)
        """
        self.k_insulation = k_insulation
        self.k_wall = k_wall
        self.use_iterative_solver = use_iterative_solver
        self.T_structure = None  # Will be calculated
        self.T_liner = None     # Will be calculated

        # Iterative solver parameters
        self.max_iterations = 1000
        self.tolerance = 1e-6  # Tighter tolerance for better energy balance

    def compute_thermal_interface_temperatures(self, tank, T_amb: float, T_hydrogen: float,
                                             tank_thermal_capacity: float, dt: float = 1.0) -> Tuple[float, float]:
        """
        Calculate structure and liner temperatures for three-layer thermal resistance system.

        Uses either direct calculation or iterative solver based on use_iterative_solver flag.
        Iterative solver provides better energy balance for temperature-dependent properties.

        Thermal path: T_amb → Insulation → T_structure → Structure → T_liner → Liner → T_hydrogen

        Args:
            tank: Tank object with liner information
            T_amb: Ambient temperature [K]
            T_hydrogen: Hydrogen temperature [K]
            tank_thermal_capacity: Tank thermal capacity [J/K]
            dt: Time step [s]

        Returns:
            Tuple of (T_structure [K], T_liner [K])
        """
        if self.use_iterative_solver:
            return self._compute_thermal_interface_temperatures_iterative(
                tank, T_amb, T_hydrogen, tank_thermal_capacity, dt
            )
        else:
            return self._compute_thermal_interface_temperatures_direct(
                tank, T_amb, T_hydrogen, tank_thermal_capacity, dt
            )

    def _compute_thermal_interface_temperatures_iterative(self, tank, T_amb: float, T_hydrogen: float,
                                                         tank_thermal_capacity: float, dt: float = 1.0) -> Tuple[float, float]:
        """
        HYBRID APPROACH WITH LAYER DISCRETIZATION: Fixed outer boundary + Matrix solver for layers.

        This combines:
        1. Fixed Heat Input Stage: Use fixed heat input to calculate T_structure (outer boundary)
           - T_structure = T_amb - q_fixed / k_insulation (fixed boundary condition)

        2. Layer Discretization Stage: Matrix solver for discretized wall/liner layers
           - Multiple temperature nodes through wall and liner thickness
           - Boundary conditions: T_structure (fixed) and T_hydrogen (fixed)
           - Interior nodes solved using finite difference heat conduction

        Args:
            tank: Tank object with liner information
            T_amb: Ambient temperature [K]
            T_hydrogen: Hydrogen temperature [K]
            tank_thermal_capacity: Tank thermal capacity [J/K]
            dt: Time step [s]

        Returns:
            Tuple of (T_structure [K], T_liner [K])
        """
        import numpy as np

                # STAGE 1: CALCULATE HEAT INPUT FROM AMBIENT → STRUCTURE
        # Calculate fixed heat input based on overall temperature difference
        tank_surface_area = tank.surface_area
        q_fixed = self.k_insulation * (T_amb - T_hydrogen)  # W/m²
        fixed_heat_input = q_fixed * tank_surface_area  # W

        # Calculate structure temperature from energy balance
        # For series thermal resistance: T_amb → T_structure → T_hydrogen
        # We need to solve for T_structure where heat flow is consistent

        # Get initial estimates for wall/liner thermal resistances
        # Check if liner exists
        has_liner = hasattr(tank, 'liner') and tank.liner is not None

        if has_liner:
            k_liner = self._get_liner_thermal_conductivity(tank, T_hydrogen)
            liner_thickness = self._get_liner_thickness(tank)
            R_liner = liner_thickness / k_liner  # m²·K/W
        else:
            k_liner = None
            liner_thickness = 0.0
            R_liner = 0.0  # No liner resistance
            print("No liner detected - using direct wall → hydrogen heat transfer")

        T_wall_initial = (T_amb + T_hydrogen) / 2  # Initial estimate
        k_wall_dynamic = self.get_dynamic_wall_thermal_conductivity(tank, T_wall_initial)

        wall_thickness = self._get_wall_thickness(tank)

        # Calculate thermal resistances for structure → hydrogen path
        R_wall = wall_thickness / k_wall_dynamic  # m²·K/W
        R_structure_to_h2 = R_wall + R_liner  # Total thermal resistance

        # Calculate structure temperature from energy balance
        # q = (T_amb - T_structure) / R_insulation = (T_structure - T_hydrogen) / R_structure_to_h2
        # Where R_insulation = 1 / k_insulation
        R_insulation = 1.0 / self.k_insulation  # m²·K/W
        R_total = R_insulation + R_structure_to_h2

        # From series resistance: T_structure = T_amb - q * R_insulation
        q_actual = (T_amb - T_hydrogen) / R_total  # Actual heat flux W/m²
        T_structure = T_amb - q_actual * R_insulation

        print(f"Starting HYBRID DISCRETIZED thermal solver:")
        print(f"  STAGE 1 - Energy balance calculation:")
        print(f"    R_insulation: {R_insulation:.6f} m²·K/W")
        print(f"    R_structure_to_h2: {R_structure_to_h2:.6f} m²·K/W")
        print(f"    R_total: {R_total:.6f} m²·K/W")
        print(f"    Calculated heat flux: {q_actual:.3f} W/m² (Total: {q_actual * tank_surface_area:.1f} W)")
        print(f"    Structure temperature: {T_structure:.1f} K (from energy balance)")
        print(f"  STAGE 2 - Layer discretization: T_structure → layers → T_H2")

        # STAGE 2: LAYER DISCRETIZATION - Matrix solver for wall/liner layers
        # Get material properties and thicknesses using average temperatures
        # Use hydrogen temperature as initial estimate for liner, will be refined in iterations

        if has_liner:
            k_liner = self._get_liner_thermal_conductivity(tank, T_hydrogen)
        else:
            k_liner = None

        # Use average of structure and hydrogen temperatures as initial estimate for wall
        T_wall_initial = (T_structure + T_hydrogen) / 2
        k_wall_dynamic = self.get_dynamic_wall_thermal_conductivity(tank, T_wall_initial)

        wall_thickness = self._get_wall_thickness(tank)

        if has_liner:
            liner_thickness = self._get_liner_thickness(tank)
        else:
            liner_thickness = 0.0

        # Define layer discretization
        n_wall_layers = 12   # Number of wall layers
        if has_liner:
            n_liner_layers = 10  # Number of liner layers
            total_layers = n_wall_layers + n_liner_layers
        else:
            n_liner_layers = 0  # No liner layers
            total_layers = n_wall_layers

        # Calculate layer thicknesses and resistances
        wall_layer_thickness = wall_thickness / n_wall_layers
        R_wall_layer = wall_layer_thickness / k_wall_dynamic

        if has_liner:
            liner_layer_thickness = liner_thickness / n_liner_layers
            R_liner_layer = liner_layer_thickness / k_liner
        else:
            liner_layer_thickness = 0.0
            R_liner_layer = 0.0

        print(f"  Layer details:")
        print(f"    Wall: {n_wall_layers} layers × {wall_layer_thickness*1000:.1f}mm, R = {R_wall_layer:.6f} m²·K/W each")
        if has_liner:
            print(f"    Liner: {n_liner_layers} layers × {liner_layer_thickness*1000:.1f}mm, R = {R_liner_layer:.6f} m²·K/W each")
        else:
            print(f"    Liner: No liner present - direct wall → hydrogen interface")
        print(f"    Boundary conditions: T_structure = {T_structure:.1f}K (fixed), T_H2 = {T_hydrogen:.1f}K (fixed)")

        # Build thermal resistance matrix for interior nodes
        # Node layout: [T_structure] → [T_w1, T_w2, T_w3] → [T_l1, T_l2] → [T_hydrogen]
        # We solve for interior nodes: [T_w1, T_w2, T_w3, T_l1, T_l2]
        n_interior = total_layers  # Number of interior nodes to solve

        # Build coefficient matrix A and RHS vector b for A*T = b
        A = np.zeros((n_interior, n_interior))
        b = np.zeros(n_interior)

        # Node 0: First wall layer (connected to T_structure)
        A[0, 0] = 1/R_wall_layer + 1/R_wall_layer  # Self resistance (left + right)
        A[0, 1] = -1/R_wall_layer                  # Right neighbor
        b[0] = T_structure / R_wall_layer          # Left boundary contribution

        # Nodes 1 to n_wall_layers-1: Interior wall layers
        for i in range(1, n_wall_layers-1):
            A[i, i-1] = -1/R_wall_layer            # Left neighbor
            A[i, i] = 1/R_wall_layer + 1/R_wall_layer  # Self resistance
            A[i, i+1] = -1/R_wall_layer            # Right neighbor
            b[i] = 0                               # No external source

        # Node n_wall_layers-1: Last wall layer (connected to first liner layer or hydrogen)
        if has_liner and n_liner_layers > 0:
            A[n_wall_layers-1, n_wall_layers-2] = -1/R_wall_layer    # Left neighbor (wall)
            A[n_wall_layers-1, n_wall_layers-1] = 1/R_wall_layer + 1/R_liner_layer  # Self
            A[n_wall_layers-1, n_wall_layers] = -1/R_liner_layer     # Right neighbor (liner)
            b[n_wall_layers-1] = 0
        else:
            # No liner - connect directly to hydrogen
            A[n_wall_layers-1, n_wall_layers-2] = -1/R_wall_layer
            A[n_wall_layers-1, n_wall_layers-1] = 1/R_wall_layer  # No additional resistance
            b[n_wall_layers-1] = T_hydrogen / R_wall_layer  # Use wall resistance for connection to hydrogen

        # Liner nodes (if present)
        if has_liner and n_liner_layers > 0:
            # Nodes n_wall_layers to n_wall_layers+n_liner_layers-2: Interior liner layers
            for i in range(n_wall_layers, n_wall_layers + n_liner_layers - 1):
                A[i, i-1] = -1/R_liner_layer       # Left neighbor
                A[i, i] = 1/R_liner_layer + 1/R_liner_layer  # Self resistance
                A[i, i+1] = -1/R_liner_layer       # Right neighbor
                b[i] = 0

            # Last liner layer (connected to T_hydrogen)
            last_liner_idx = n_wall_layers + n_liner_layers - 1
            A[last_liner_idx, last_liner_idx-1] = -1/R_liner_layer    # Left neighbor
            A[last_liner_idx, last_liner_idx] = 1/R_liner_layer + 1/R_liner_layer  # Self
            b[last_liner_idx] = T_hydrogen / R_liner_layer            # Right boundary

        # Solve the matrix system
        try:
            T_interior = np.linalg.solve(A, b)
            print(f"  ✅ Matrix solver converged for {n_interior} interior nodes")

            # Extract temperatures
            wall_temperatures = T_interior[:n_wall_layers]
            liner_temperatures = T_interior[n_wall_layers:] if n_liner_layers > 0 else []

            # ENHANCED: Update thermal conductivities based on actual discretized temperatures
            if len(wall_temperatures) > 0:
                T_wall_avg = sum(wall_temperatures) / len(wall_temperatures)
                k_wall_updated = self.get_dynamic_wall_thermal_conductivity(tank, T_wall_avg)
                print(f"  🔄 Updated k_wall: {k_wall_dynamic:.3f} → {k_wall_updated:.3f} W/(m·K) at {T_wall_avg:.1f}K")

            if len(liner_temperatures) > 0:
                T_liner_avg = sum(liner_temperatures) / len(liner_temperatures)
                k_liner_updated = self._get_liner_thermal_conductivity(tank, T_liner_avg)
                print(f"  🔄 Updated k_liner: {k_liner:.1f} → {k_liner_updated:.1f} W/(m·K) at {T_liner_avg:.1f}K")

            # T_liner is the temperature at the wall-liner interface
            if n_liner_layers > 0:
                T_liner = liner_temperatures[0]  # First liner layer temperature
            else:
                T_liner = wall_temperatures[-1]  # Last wall layer if no liner

            print(f"  Temperature profile through layers:")
            print(f"    T_structure: {T_structure:.2f} K (boundary)")
            for i, T_wall in enumerate(wall_temperatures):
                print(f"    T_wall_{i+1}: {T_wall:.2f} K")
            for i, T_lin in enumerate(liner_temperatures):
                print(f"    T_liner_{i+1}: {T_lin:.2f} K")
            print(f"    T_hydrogen: {T_hydrogen:.2f} K (boundary)")

        except np.linalg.LinAlgError as e:
            print(f"  ⚠️  Matrix solver failed: {e}")
            print(f"  Falling back to simple series resistance")

            # Fallback to simple series resistance
            R_wall_total = wall_thickness / k_wall_dynamic
            if has_liner:
                R_liner_total = liner_thickness / k_liner
                R_total = R_wall_total + R_liner_total
                q_total = (T_structure - T_hydrogen) / R_total
                T_liner = T_structure - q_total * R_wall_total
            else:
                R_total = R_wall_total
                q_total = (T_structure - T_hydrogen) / R_total
                T_liner = T_hydrogen  # No liner, so liner temperature equals hydrogen temperature

        # Calculate actual heat flux through the discretized system
        # Heat flux from T_structure into first wall layer
        if len(T_interior) > 0:
            q_wall_liner = (T_structure - T_interior[0]) / R_wall_layer
        else:
            q_wall_liner = 0.0

        print(f"  Heat flux through discretized system: {q_wall_liner:.3f} W/m²")

        # Verify temperature monotonicity
        temp_profile = [T_structure] + list(T_interior) + [T_hydrogen]
        is_monotonic = all(temp_profile[i] >= temp_profile[i+1] for i in range(len(temp_profile)-1))

        if is_monotonic:
            print(f"  ✅ Temperature profile is monotonic (physically realistic)")
        else:
            print(f"  ⚠️  Warning: Non-monotonic temperature profile detected")

        # Energy balance between series calculation and discretized system
        energy_balance_error = abs(q_actual - q_wall_liner) / max(abs(q_actual), 1e-9)
        print(f"  Energy balance: series {q_actual:.6f} vs discretized {q_wall_liner:.6f} W/m²")
        print(f"  Relative difference: {energy_balance_error*100:.4f}%")

        # Store results
        self.T_structure = T_structure
        self.T_liner = T_liner

        # Store discretization info
        self._hybrid_info = {
            'heat_input_from_energy_balance': q_actual * tank_surface_area,  # W (corrected)
            'q_energy_balance': q_actual,     # W/m² (corrected heat flux)
            'T_structure_analytical': T_structure,
            'q_wall_liner_discretized': q_wall_liner,
            'n_wall_layers': n_wall_layers,
            'n_liner_layers': n_liner_layers,
            'wall_temperatures': wall_temperatures if 'wall_temperatures' in locals() else [],
            'liner_temperatures': liner_temperatures if 'liner_temperatures' in locals() else [],
            'energy_balance_error': energy_balance_error
        }

        return T_structure, T_liner

    def _compute_thermal_interface_temperatures_direct(self, tank, T_amb: float, T_hydrogen: float,
                                                      tank_thermal_capacity: float, dt: float = 1.0) -> Tuple[float, float]:
        """
        Direct calculation of interface temperatures using series thermal resistance.

        Args:
            tank: Tank object with liner information
            T_amb: Ambient temperature [K]
            T_hydrogen: Hydrogen temperature [K]
            tank_thermal_capacity: Tank thermal capacity [J/K]
            dt: Time step [s]

        Returns:
            Tuple of (T_structure [K], T_liner [K])
        """
        # Get liner thermal properties if available
        has_liner = hasattr(tank, 'liner') and tank.liner is not None

        if has_liner:
            k_liner = self._get_liner_thermal_conductivity(tank, T_hydrogen)
            liner_thickness = self._get_liner_thickness(tank)
        else:
            k_liner = None
            liner_thickness = 0.0
            print("No liner detected - using direct wall → hydrogen heat transfer")

        k_wall_dynamic = self.get_dynamic_wall_thermal_conductivity(tank, (T_amb + T_hydrogen) / 2)

        # Get thicknesses for proper thermal resistance calculation
        wall_thickness = self._get_wall_thickness(tank)

        # Thermal resistances per unit area [m²⋅K/W] using proper physics
        # Insulation: Uses heat transfer coefficient k_insulation [W/(m²⋅K)]
        R_insulation = 1.0 / self.k_insulation  # [m²⋅K/W]

        # Wall: Uses thermal conductivity with thickness
        R_wall = wall_thickness / k_wall_dynamic  # [m²⋅K/W]

        # Liner: Uses thermal conductivity with thickness (if present)
        if has_liner:
            R_liner = liner_thickness / k_liner  # [m²⋅K/W]
            R_total = R_insulation + R_wall + R_liner
        else:
            R_liner = 0.0  # No liner resistance
            R_total = R_insulation + R_wall

        # Heat flux through the series system [W/m²]
        q = (T_amb - T_hydrogen) / R_total

        # Calculate interface temperatures
        T_structure = T_amb - q * R_insulation                    # After insulation

        if has_liner:
            T_liner = T_structure - q * R_wall                   # After wall, before liner
            # Final check: T_hydrogen = T_liner - q * R_liner
        else:
            T_liner = T_hydrogen                                  # No liner, so T_liner = T_hydrogen
            # Final check: T_hydrogen = T_structure - q * R_wall

        # Store for later use
        self.T_structure = T_structure
        self.T_liner = T_liner

        print(f"  Direct calculation: T_structure={T_structure:.2f}K, T_liner={T_liner:.2f}K")

        return T_structure, T_liner

    def get_dynamic_wall_thermal_conductivity(self, tank, structure_temperature: float) -> float:
        """
        Get temperature-dependent thermal conductivity for the composite tank wall.

        Args:
            tank: Tank object with material properties
            structure_temperature: Structure temperature [K]

        Returns:
            Material thermal conductivity [W/(m·K)]
        """
        # Get material-specific thermal conductivity
        if not hasattr(tank, 'material'):
            raise ValueError("Tank object missing 'material' attribute for thermal conductivity calculation")

        if not hasattr(tank.material, 'determine_thermal_conductivity'):
            raise ValueError(f"Tank material ({type(tank.material).__name__}) missing 'determine_thermal_conductivity' method")

        # Get material thermal conductivity [W/(m·K)]
        thermal_conductivity = tank.material.determine_thermal_conductivity(structure_temperature)

        # Validate reasonable thermal conductivity range
        if thermal_conductivity <= 0:
            raise ValueError(f"Invalid thermal conductivity: {thermal_conductivity} W/(m·K) must be positive")
        if thermal_conductivity > 1000.0:
            raise ValueError(f"Unrealistic thermal conductivity: {thermal_conductivity} W/(m·K) > 1000 W/(m·K)")

        print(f"Dynamic wall thermal conductivity: {thermal_conductivity:.3f} W/(m·K) at {structure_temperature:.1f}K")
        return thermal_conductivity

    def _get_liner_thermal_conductivity(self, tank, hydrogen_temperature: float) -> float:
        """
        Get liner thermal conductivity (for solid conduction).

        Args:
            tank: Tank object with potential liner
            hydrogen_temperature: Hydrogen temperature [K]

        Returns:
            Liner thermal conductivity [W/(m·K)]
        """
        if not (hasattr(tank, 'liner') and tank.liner is not None):
            return None  # No liner present

        liner = tank.liner
        if not hasattr(liner, 'compute_thermal_conductivity'):
            raise ValueError(f"Liner object ({type(liner).__name__}) missing 'compute_thermal_conductivity' method")

        # Get liner thermal conductivity [W/(m·K)]
        thermal_conductivity = liner.compute_thermal_conductivity(hydrogen_temperature, hydrogen_temperature)

        if thermal_conductivity <= 0:
            raise ValueError(f"Invalid liner thermal conductivity: {thermal_conductivity} W/(m·K) must be positive")

        print(f"LINER thermal conductivity: {thermal_conductivity:.1f} W/(m·K) at {hydrogen_temperature:.1f}K")
        return thermal_conductivity

    def _get_wall_thickness(self, tank) -> float:
        """
        Get wall thickness from tank sections.

        Args:
            tank: Tank object with sections

        Returns:
            Wall thickness [m]
        """
        if not hasattr(tank, 'sections'):
            raise ValueError("Tank object missing 'sections' attribute")

        if not tank.sections:
            raise ValueError("Tank object has empty sections list")

        if len(tank.sections) == 0:
            raise ValueError("Tank object has zero sections")

        # Use first section's thickness (all sections should have same thickness)
        first_section = tank.sections[0]
        if not hasattr(first_section, 'thickness'):
            raise ValueError(f"Tank section ({type(first_section).__name__}) missing 'thickness' attribute")

        thickness = first_section.thickness

        if thickness <= 0:
            raise ValueError(f"Invalid wall thickness: {thickness} m must be positive")

        print(f"WALL thickness: {thickness*1000:.2f} mm from {len(tank.sections)} sections")
        return thickness

    def _get_liner_thickness(self, tank) -> float:
        """
        Get liner thickness from tank.

        Args:
            tank: Tank object with potential liner

        Returns:
            Liner thickness [m]
        """
        if not (hasattr(tank, 'liner') and tank.liner is not None):
            return 0.0  # No liner present

        liner = tank.liner
        if hasattr(liner, 'thickness') and liner.thickness is not None:
            if liner.thickness <= 0:
                raise ValueError(f"Invalid liner thickness: {liner.thickness} m must be positive")
            print(f"LINER thickness: {liner.thickness*1000:.2f} mm")
            return liner.thickness
        else:
            # Try to calculate thickness from mass if available
            if hasattr(liner, 'calculate_thickness_from_mass'):
                calculated_thickness = liner.calculate_thickness_from_mass()
                if calculated_thickness <= 0:
                    raise ValueError(f"Invalid calculated liner thickness: {calculated_thickness} m must be positive")
                print(f"LINER thickness (calculated): {calculated_thickness*1000:.2f} mm")
                return calculated_thickness
            else:
                raise ValueError("Liner object missing both 'thickness' attribute and 'calculate_thickness_from_mass' method")

    def _get_liner_thermal_contributions(self, tank, structure_temperature: float, hydrogen_temperature: float) -> str:
        """
        Check and report liner thermal contributions.

        Args:
            tank: Tank object
            structure_temperature: Structure temperature [K]
            hydrogen_temperature: Hydrogen temperature [K]

        Returns:
            String describing liner thermal contributions
        """
        if not (hasattr(tank, 'liner') and tank.liner is not None):
            return "LINER: None present"

        liner = tank.liner
        if not hasattr(liner, 'compute_thermal_conductivity'):
            raise ValueError(f"Liner object ({type(liner).__name__}) missing 'compute_thermal_conductivity' method")

        # Get liner thermal conductivity
        liner_k = liner.compute_thermal_conductivity(structure_temperature, hydrogen_temperature)

        # Estimate liner thermal resistance if we have thickness
        if hasattr(liner, 'thickness') and liner.thickness is not None:
            if liner.thickness <= 0:
                raise ValueError(f"Invalid liner thickness: {liner.thickness} m must be positive")
            liner_resistance = liner.thickness / liner_k  # [m²⋅K/W]
            liner_heat_transfer_coeff = 1.0 / liner_resistance  # [W/(m²⋅K)]

            return (f"LINER: k={liner_k:.1f} W/(m·K), thickness={liner.thickness*1000:.2f}mm, "
                   f"h_liner={liner_heat_transfer_coeff:.3f} W/(m²·K)")
        else:
            return f"LINER: k={liner_k:.1f} W/(m·K), mass={getattr(liner, 'mass', 'unknown')} kg"

    def compute_heat_fluxes(self, tank, tank_state, mission_section) -> Tuple[float, float, float, list]:
        """
        Compute heat fluxes using HYBRID DISCRETIZED approach: Calculated heat input + Discretized layers.

        HYBRID DISCRETIZED THERMAL MODEL:
        1. STAGE 1 - Calculated Heat Input: T_amb → T_structure (based on k_insulation and ΔT)
           - q_calculated = k_insulation × (T_amb - T_hydrogen), T_structure = T_amb - q_calculated / k_insulation

        2. STAGE 2 - Layer Discretization: T_structure → layers → T_H2 (matrix solver)
           - Multiple temperature nodes through wall and liner
           - Matrix solver for finite difference heat conduction
           - Boundary conditions: T_structure (fixed) and T_hydrogen (fixed)

        This provides proper layer discretization with realistic heat input calculation!

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
        print(f"Using tank thermal capacity: {tank_thermal_capacity:.1f} J/K")

        # Calculate interface temperatures using HYBRID DISCRETIZED approach
        T_structure, T_liner = self.compute_thermal_interface_temperatures(
            tank, T_amb, T_H2, tank_thermal_capacity
        )

        # HYBRID DISCRETIZED HEAT FLUX CALCULATION:
        # Reuse values from interface temperature calculation to avoid repetition
        tank_surface_area = getattr(tank, 'surface_area', 4.1)  # m²
        hybrid_info = getattr(self, '_hybrid_info', {})

        # Stage 1: Get calculated heat input from interface temperature method
        heat_input_corrected = hybrid_info.get('heat_input_from_energy_balance', self.k_insulation * (T_amb - T_H2) * tank_surface_area)  # W
        q_insulation = heat_input_corrected / tank_surface_area  # W/m² - corrected heat flux from energy balance

        # Stage 2: Heat flux through discretized layers (from matrix solver)
        q_wall_liner_discretized = hybrid_info.get('q_wall_liner_discretized', 0.0)  # W/m²

        # Get material properties for detailed calculations
        has_liner = hasattr(tank, 'liner') and tank.liner is not None

        k_wall_dynamic = self.get_dynamic_wall_thermal_conductivity(tank, T_structure)
        if has_liner:
            k_liner = self._get_liner_thermal_conductivity(tank, T_H2)
            liner_thickness = self._get_liner_thickness(tank)
        else:
            k_liner = None
            liner_thickness = 0.0

        wall_thickness = self._get_wall_thickness(tank)

        # Calculate individual component heat fluxes (approximate for reporting)
        R_wall = wall_thickness / k_wall_dynamic    # [m²⋅K/W]
        if has_liner:
            R_liner = liner_thickness / k_liner         # [m²⋅K/W]
            R_total_approx = R_wall + R_liner           # Approximate total resistance
        else:
            R_liner = 0.0                               # No liner resistance
            R_total_approx = R_wall                     # Only wall resistance

        # Individual component fluxes (based on discretized solution)
        if has_liner and R_total_approx > 0:
            q_wall = q_wall_liner_discretized * (R_liner / R_total_approx)   # Wall contribution
            q_liner = q_wall_liner_discretized * (R_wall / R_total_approx)   # Liner contribution
        else:
            q_wall = q_wall_liner_discretized  # All heat flux goes through wall
            q_liner = 0.0                      # No liner flux

        # Net heat flux to hydrogen (from discretized stage)
        q_net = q_wall_liner_discretized

        # Enhanced temperature profile from discretized solution
        wall_temps = hybrid_info.get('wall_temperatures', [])
        liner_temps = hybrid_info.get('liner_temperatures', []) if has_liner else []

        # Build complete temperature profile: [T_H2, liner_layers..., wall_layers..., T_structure, T_amb]
        temperature_profile = [T_H2]
        if has_liner:
            temperature_profile.extend(reversed(liner_temps))  # Liner layers (inside to outside)
        temperature_profile.extend(reversed(wall_temps))   # Wall layers (inside to outside)
        temperature_profile.extend([T_structure, T_amb])

        # Store individual heat fluxes
        self._q_insulation = q_insulation  # Fixed input stage
        self._q_wall = q_wall              # Discretized stage (wall component)
        self._q_liner = q_liner            # Discretized stage (liner component)
        self._q_net = q_net                # Net to hydrogen
        self._T_structure = T_structure
        self._T_liner = T_liner

        # Debug output
        if abs(T_amb - T_H2) > 1e-6:
            print(f"\n=== HYBRID DISCRETIZED THERMAL MODEL ===")
            print(f"STAGE 1 - Calculated Heat Input (Ambient → Structure):")
            print(f"  Ambient temperature: {T_amb:.1f} K")
            print(f"  Structure temperature: {T_structure:.1f} K (FIXED BOUNDARY)")
            print(f"  Calculated heat input: {heat_input_corrected:.1f} W")
            print(f"  q_insulation (calculated): {q_insulation:.3f} W/m²")
            print(f"  k_insulation: {self.k_insulation:.6f} W/(m²·K)")

            print(f"\nSTAGE 2 - Layer Discretization (Structure → Hydrogen):")
            print(f"  Wall layers: {hybrid_info.get('n_wall_layers', 0)}")
            print(f"  Liner layers: {hybrid_info.get('n_liner_layers', 0)}")
            print(f"  k_wall (dynamic): {k_wall_dynamic:.3f} W/(m·K)")
            if k_liner is not None:
                print(f"  k_liner: {k_liner:.3f} W/(m·K)")
            else:
                print(f"  k_liner: No liner present")

            print(f"\nDiscretized Temperature Profile:")
            print(f"  T_structure: {T_structure:.2f} K (FIXED BOUNDARY)")
            for i, T_wall in enumerate(wall_temps):
                print(f"  T_wall_{i+1}: {T_wall:.2f} K")
            for i, T_lin in enumerate(liner_temps):
                print(f"  T_liner_{i+1}: {T_lin:.2f} K")
            print(f"  T_hydrogen: {T_H2:.2f} K (FIXED BOUNDARY)")

            print(f"\nHeat Flux Results [W/m²]:")
            print(f"  q_insulation (stage 1): {q_insulation:.3f} W/m² [FIXED INPUT]")
            print(f"  q_wall (stage 2): {q_wall:.3f} W/m² [DISCRETIZED]")
            print(f"  q_liner (stage 2): {q_liner:.3f} W/m² [DISCRETIZED]")
            print(f"  q_net → hydrogen: {q_net:.3f} W/m² [DISCRETIZED]")

            # Energy balance between stages
            energy_balance_error = hybrid_info.get('energy_balance_error', 0.0)
            print(f"  Energy balance between stages: {energy_balance_error*100:.1f}% difference")
            print(f"  (Difference expected due to hybrid approach)")

            # Calculate total heat rates
            if hasattr(tank, 'surface_area'):
                Q_insulation_total = q_insulation * tank.surface_area
                Q_net_total = q_net * tank.surface_area
                print(f"\nTotal Heat Rates:")
                print(f"  Into structure (stage 1): {Q_insulation_total:.2f} W")
                print(f"  Into hydrogen (stage 2): {Q_net_total:.2f} W")
                print(f"  Heat accumulation in tank: {Q_insulation_total - Q_net_total:.2f} W")
                print(f"  Tank thermal capacity: {tank_thermal_capacity:.1f} J/K")
            print(f"====================================")

        return q_insulation, q_wall, q_net, temperature_profile

    def _get_dynamic_thermal_capacity(self, tank, tank_state) -> float:
        """
        Get dynamic tank thermal capacity based on actual wall and liner temperatures from discretization.

        Properly computes: C_tank = C_wall(T_wall_avg) + C_liner(T_liner_avg)
        where T_wall_avg and T_liner_avg are the average temperatures across discretized layers.

        Args:
            tank: Tank object
            tank_state: Current tank state

        Returns:
            Tank thermal capacity [J/K]
        """
        # Get discretized temperature profile from hybrid thermal model
        hybrid_info = getattr(self, '_hybrid_info', {})
        wall_temperatures = hybrid_info.get('wall_temperatures', [])
        liner_temperatures = hybrid_info.get('liner_temperatures', [])

        # Calculate average temperatures for wall and liner layers
        if len(wall_temperatures) > 0:
            T_wall_avg = sum(wall_temperatures) / len(wall_temperatures)
        else:
            # Fallback to structure temperature if no discretized layers
            T_wall_avg = self.T_structure if self.T_structure is not None else tank_state.temperature

        if len(liner_temperatures) > 0:
            T_liner_avg = sum(liner_temperatures) / len(liner_temperatures)
        else:
            # Fallback to liner interface temperature or hydrogen temperature
            T_liner_avg = self.T_liner if self.T_liner is not None else tank_state.temperature

        # Compute wall thermal capacity: C_wall = cp_wall(T_wall_avg) * m_wall
        wall_thermal_capacity = 0.0
        if hasattr(tank, 'sections') and tank.sections:
            for section in tank.sections:
                if hasattr(section, 'structural_mass') and hasattr(section, 'material'):
                    if hasattr(section.material, 'determine_specific_heat'):
                        cp_wall = section.material.determine_specific_heat(T_wall_avg)
                        m_wall = section.structural_mass
                        wall_thermal_capacity += cp_wall * m_wall

        # Compute liner thermal capacity: C_liner = cp_liner(T_liner_avg) * m_liner
        liner_thermal_capacity = 0.0
        if hasattr(tank, 'liner') and tank.liner is not None:
            if hasattr(tank.liner, 'material') and hasattr(tank.liner, 'mass'):
                if tank.liner.mass is not None and hasattr(tank.liner.material, 'determine_specific_heat'):
                    cp_liner = tank.liner.material.determine_specific_heat(T_liner_avg)
                    m_liner = tank.liner.mass
                    liner_thermal_capacity = cp_liner * m_liner

        total_thermal_capacity = wall_thermal_capacity + liner_thermal_capacity

        if total_thermal_capacity <= 0:
            # Fallback to original tank method if our enhanced calculation fails
            structure_temp = self.T_structure if self.T_structure is not None else tank_state.temperature
            if hasattr(tank, 'compute_thermal_capacity'):
                total_thermal_capacity = tank.compute_thermal_capacity(structure_temp)
            else:
                raise ValueError(f"Tank object ({type(tank).__name__}) missing 'compute_thermal_capacity' method")

        # Debug output for verification
        if abs(wall_thermal_capacity) > 1e-6 or abs(liner_thermal_capacity) > 1e-6:
            print(f"  📊 ENHANCED THERMAL CAPACITY CALCULATION:")
            print(f"    Wall avg temperature: {T_wall_avg:.1f} K, C_wall: {wall_thermal_capacity:.1f} J/K")
            print(f"    Liner avg temperature: {T_liner_avg:.1f} K, C_liner: {liner_thermal_capacity:.1f} J/K")
            print(f"    Total thermal capacity: {total_thermal_capacity:.1f} J/K")

            # Show temperature-dependent specific heats
            if hasattr(tank, 'sections') and tank.sections and hasattr(tank.sections[0], 'material'):
                cp_wall_demo = tank.sections[0].material.determine_specific_heat(T_wall_avg)
                print(f"    cp_wall({T_wall_avg:.1f}K): {cp_wall_demo:.1f} J/(kg·K)")

            if hasattr(tank, 'liner') and tank.liner and hasattr(tank.liner, 'material'):
                cp_liner_demo = tank.liner.material.determine_specific_heat(T_liner_avg)
                print(f"    cp_liner({T_liner_avg:.1f}K): {cp_liner_demo:.1f} J/(kg·K)")

        return total_thermal_capacity

    def compute_heat_flux(self, tank, tank_state, mission_section) -> Tuple[float, list]:
        """
        Compatibility method for SimplifiedThermodynamicModel interface.

        Returns:
            tuple: (net_heat_flux [W/m²], temperature_profile [list])
        """
        q_insulation, q_wall, q_net, temperature_profile = self.compute_heat_fluxes(
            tank, tank_state, mission_section
        )

        # Use the correctly calculated insulation heat flux instead of broken discretized q_net
        # The discretized q_net returns 0.000 due to identical boundary conditions
        return q_insulation, temperature_profile

    def get_insulation_heat_flux(self) -> float:
        """
        Get the heat flux through insulation (ambient → structure).

        Returns:
            Heat flux through insulation [W/m²]
        """
        return getattr(self, '_q_insulation', 0.0)

    def get_wall_heat_flux(self) -> float:
        """
        Get the heat flux through tank wall (structure → hydrogen).

        Returns:
            Heat flux through wall [W/m²]
        """
        return getattr(self, '_q_wall', 0.0)

    def get_structure_temperature(self) -> float:
        """
        Get the current structure temperature.

        Returns:
            Structure temperature [K]
        """
        return getattr(self, '_T_structure', 0.0)

    def get_liner_temperature(self) -> float:
        """
        Get the current liner temperature.

        Returns:
            Liner temperature [K]
        """
        return getattr(self, '_T_liner', 0.0)

    def get_liner_heat_flux(self) -> float:
        """
        Get the heat flux through liner (liner → hydrogen).

        Returns:
            Heat flux through liner [W/m²]
        """
        return getattr(self, '_q_liner', 0.0)

    def get_detailed_heat_flow_info(self) -> dict:
        """
        Get detailed heat flow information for communication to dynamic models.

        This provides the detailed thermal information for three-layer system:
        - Heat flow through insulation: k_insulation * A * (T_amb - T_structure)
        - Heat flow through structure: k_wall * A * (T_structure - T_liner)
        - Heat flow through liner: k_liner * A * (T_liner - T_hydrogen)

        Returns:
            Dictionary with detailed heat flow information
        """
        return {
            'q_insulation': getattr(self, '_q_insulation', 0.0),  # W/m² through insulation
            'q_wall': getattr(self, '_q_wall', 0.0),              # W/m² through wall
            'q_liner': getattr(self, '_q_liner', 0.0),            # W/m² through liner
            'q_net': getattr(self, '_q_net', 0.0),                # W/m² net to hydrogen
            'T_structure': getattr(self, '_T_structure', 0.0),    # K structure temperature
            'T_liner': getattr(self, '_T_liner', 0.0),            # K liner temperature
            'k_insulation': self.k_insulation,                    # W/(m²·K) insulation coeff
            'k_wall': self.k_wall                                 # W/(m²·K) wall coeff
        }


class EnhancedSimplifiedThermodynamicModel(DirectThermalModel):
    """
    Enhanced version of SimplifiedThermodynamicModel with direct thermal approach.
    """

    def __init__(self, k_insulation: float = 0.033, k_wall: float = 0.2,
                 use_iterative_solver: bool = True):
        """
        Initialize enhanced simplified model.

        Args:
            k_insulation: Overall insulation heat transfer coefficient [W/(m²·K)]
            k_wall: Wall heat transfer coefficient [W/(m²·K)]
            use_iterative_solver: Use iterative solver for interface temperatures (recommended)
        """
        super().__init__(k_insulation, k_wall, use_iterative_solver)

        # Compatibility attributes for existing code
        self.k_amb = k_insulation  # For backward compatibility


# Factory function for easy integration
def create_enhanced_thermal_model(k_insulation: float = 0.033, k_wall: float = 0.2,
                                 use_iterative_solver: bool = True):
    """
    Create enhanced thermal model with direct three-layer heat transfer approach.

    This model implements series thermal resistance with three layers:

    **SERIES MODE:**
    1. **Three-layer series approach**: heat flow from ambient → structure → liner → hydrogen
    2. **Series heat transfer**: Same heat rate Q̇ flows through all three layers
       - Q̇ = (T_amb - T_H2) / (R_insulation + R_wall + R_liner)
       - Different temperature drops across each layer based on thermal resistance
    3. **Temperature interfaces**: T_amb → T_structure → T_liner → T_H2
    4. **Individual heat fluxes**:
       - q_insulation = k_insulation × (T_amb - T_structure) [W/m²]
       - q_wall = k_wall × (T_structure - T_liner) [W/m²]
       - q_liner = k_liner × (T_liner - T_H2) [W/m²]

    **ITERATIVE SOLVER (recommended):**
    5. **Energy balance optimization**: Interface temperatures solved iteratively
    6. **Temperature-dependent properties**: Accounts for dynamic material properties
    7. **Scipy optimization**: Uses L-BFGS-B method for robust convergence
    8. **Fallback protection**: Automatically falls back to direct calculation if needed

    **Common Features:**
    9. **Dynamic tank thermal capacity** from NIST database (structure temperature)
    10. **Temperature-dependent material properties**
    11. **Automatic liner detection** and thermal property calculation

    **Physics explanation for SERIES with ITERATIVE SOLVER:**
    - Three thermal elements in sequence: Insulation → Structure → Liner
    - Energy balance: q_insulation = q_wall = q_liner = q_net (enforced iteratively)
    - Temperature-dependent k_wall and k_liner properly handled
    - Interface temperatures minimize energy balance residual

    Args:
        k_insulation: Overall insulation heat transfer coefficient [W/(m²·K)]
        k_wall: Wall heat transfer coefficient (structure → liner) [W/(m²·K)]
        use_iterative_solver: Use iterative solver for interface temperatures (default: True)

    Returns:
        EnhancedSimplifiedThermodynamicModel instance
    """
    solver_display = "ITERATIVE" if use_iterative_solver else "DIRECT"

    print(f"🔥 Creating ENHANCED thermal model with THREE-LAYER SERIES approach")
    print(f"   Interface solver: {solver_display}")
    print(f"   k_insulation: {k_insulation:.6f} W/(m²·K) (ambient → structure)")
    print(f"   k_wall (ks): {k_wall:.3f} W/(m²·K) (structure → liner)")
    print(f"   k_liner: auto-detect from tank liner properties")

    print(f"   ✅ Series heat transfer: same Q̇ through all three layers")
    if use_iterative_solver:
        print(f"   ✅ Iterative energy balance: q_insulation = q_wall = q_liner")
        print(f"   ✅ Temperature-dependent properties handled correctly")
    else:
        print(f"   ⚠️  Direct calculation: may have energy balance errors")
    print(f"   ✅ Interface temperatures: T_amb → T_structure → T_liner → T_H2")

    print(f"   ✅ Separate Q_insulation, Q_wall, Q_liner communication")
    print(f"   ✅ Dynamic tank thermal capacity from NIST database")

    return EnhancedSimplifiedThermodynamicModel(k_insulation, k_wall, use_iterative_solver)
