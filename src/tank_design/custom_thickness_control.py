"""
Custom thickness control for NIST G10 tanks.

This module ensures that NIST G10 tank thickness remains constant across
all time steps and scenarios by overriding the dynamic thickness calculation.
"""

import logging
from typing import Dict, Any


class CustomThicknessStructuralModel:
    """
    A structural model that always returns a fixed thickness regardless of pressure.
    
    This model overrides the default structural analysis to enforce a constant
    thickness for custom tank designs.
    """
    
    def __init__(self, fixed_thickness: float):
        """
        Initialize with a fixed thickness.
        
        Args:
            fixed_thickness: The thickness to always return [m]
        """
        self.fixed_thickness = fixed_thickness
        self.logger = logging.getLogger(__name__)
        
    def compute_thickness(self, section, pressure):
        """
        Compute thickness (always returns the fixed value).
        
        Args:
            section: Tank section (ignored)
            pressure: Operating pressure (ignored)
            
        Returns:
            Fixed thickness value [m]
        """
        return self.fixed_thickness
        
    def __str__(self):
        return f"CustomThicknessStructuralModel(thickness={self.fixed_thickness:.4f}m)"


class ThicknessController:
    """
    Controller to maintain constant tank section thicknesses.
    """
    
    def __init__(self, fixed_thicknesses: Dict[str, float] = None):
        """
        Initialize thickness controller.
        
        Args:
            fixed_thicknesses: Dict mapping section types to fixed thicknesses [m]
                              If None, will be determined from first tank analysis
        """
        self.fixed_thicknesses = fixed_thicknesses or {}
        self.is_initialized = False
        self.logger = logging.getLogger(__name__)
        
    def capture_reference_thicknesses(self, tank) -> Dict[str, float]:
        """
        Capture thickness values from tank sections as reference.
        
        Args:
            tank: Tank object with sections
            
        Returns:
            Dict mapping section types to thicknesses
        """
        thicknesses = {}
        
        for i, section in enumerate(tank.sections):
            section_type = type(section).__name__
            try:
                thickness = section.thickness
                key = f"{section_type}_{i}"  # Include index to handle multiple sections of same type
                thicknesses[key] = thickness
                self.logger.info(f"  Captured {key}: {thickness:.6f} m")
            except Exception as e:
                self.logger.warning(f"  Could not capture thickness for {section_type}: {e}")
                
        return thicknesses
        
    def apply_fixed_thicknesses(self, tank):
        """
        Apply fixed thicknesses to tank sections.
        
        Args:
            tank: Tank object to modify
            
        Returns:
            Modified tank object
        """
        if not self.is_initialized:
            # First time - capture reference thicknesses
            self.fixed_thicknesses = self.capture_reference_thicknesses(tank)
            self.is_initialized = True
            self.logger.info(f"✅ Thickness controller initialized with {len(self.fixed_thicknesses)} sections")
        
        # Apply fixed thicknesses by overriding structural model
        for i, section in enumerate(tank.sections):
            section_type = type(section).__name__
            key = f"{section_type}_{i}"
            
            if key in self.fixed_thicknesses:
                fixed_thickness = self.fixed_thicknesses[key]
                
                # Create a mock structural model that returns fixed thickness
                class FixedThicknessStructuralModel:
                    def __init__(self, thickness):
                        self.fixed_thickness = thickness
                        
                    def compute_thickness(self, section, pressure):
                        return self.fixed_thickness
                
                # Override the structural model
                section.structural_model = FixedThicknessStructuralModel(fixed_thickness)
                
                # Verify the change
                new_thickness = section.thickness
                if abs(new_thickness - fixed_thickness) < 1e-10:
                    self.logger.debug(f"  ✅ Applied fixed thickness to {key}: {fixed_thickness:.6f} m")
                else:
                    self.logger.warning(f"  ⚠️  Thickness mismatch for {key}: expected {fixed_thickness:.6f}, got {new_thickness:.6f}")
            else:
                self.logger.warning(f"  ⚠️  No fixed thickness available for {key}")
        
        return tank
    
    def get_effective_thickness(self, tank) -> float:
        """
        Get effective tank thickness (average of all sections, weighted by area).
        
        Args:
            tank: Tank object
            
        Returns:
            Effective thickness [m]
        """
        if not hasattr(tank, 'sections') or not tank.sections:
            return 0.0
            
        total_area = 0.0
        weighted_thickness = 0.0
        
        for section in tank.sections:
            try:
                area = section.surface_area
                thickness = section.thickness
                weighted_thickness += area * thickness
                total_area += area
            except Exception as e:
                self.logger.warning(f"Could not get area/thickness for section {type(section).__name__}: {e}")
        
        if total_area > 0:
            return weighted_thickness / total_area
        else:
            return 0.0
            
    def verify_thickness_consistency(self, tank, tolerance: float = 1e-10) -> bool:
        """
        Verify that all tank section thicknesses match expected fixed values.
        
        Args:
            tank: Tank object to verify
            tolerance: Acceptable thickness variation [m]
            
        Returns:
            True if all thicknesses are consistent
        """
        if not self.is_initialized:
            self.logger.warning("Thickness controller not initialized")
            return False
            
        all_consistent = True
        
        for i, section in enumerate(tank.sections):
            section_type = type(section).__name__
            key = f"{section_type}_{i}"
            
            if key in self.fixed_thicknesses:
                expected = self.fixed_thicknesses[key]
                actual = section.thickness
                deviation = abs(actual - expected)
                
                if deviation > tolerance:
                    self.logger.error(f"❌ Thickness inconsistency in {key}: expected {expected:.6f}, got {actual:.6f} (deviation: {deviation:.8f})")
                    all_consistent = False
                else:
                    self.logger.debug(f"✅ Thickness consistent for {key}: {actual:.6f} m")
            else:
                self.logger.warning(f"⚠️  No reference thickness for {key}")
                all_consistent = False
        
        return all_consistent
