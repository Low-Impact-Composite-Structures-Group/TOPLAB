import numpy as np
import matplotlib.pyplot as plt


# Aluminum 5083 material property coefficients from NIST data
# Equation: y = 10^(a + b*log10(T) + c*(log10(T))^2 + d*(log10(T))^3 + e*(log10(T))^4 + f*(log10(T))^5 + g*(log10(T))^6 + h*(log10(T))^7 + i*(log10(T))^8)
# Where T is temperature in K, and y is the property value

# Thermal Conductivity - W/(m·K)
THERMAL_CONDUCTIVITY_COEFFS = {
	'a': -0.90933,
	'b': 5.751,
	'c': -11.112,
	'd': 13.612,
	'e': -9.3977,
	'f': 3.6873,
	'g': -0.77295,
	'h': 0.067336,
	'i': 0,
	'data_range': (4, 300),
	'equation_range': (1, 300),
	'units': 'W/(m·K)',
	'fit_error': 1  # % error relative to data
}

# Specific Heat - J/(kg·K)
SPECIFIC_HEAT_COEFFS = {
	'a': 46.6467,
	'b': -314.292,
	'c': 866.662,
	'd': -1298.3,
	'e': 1162.27,
	'f': -637.795,
	'g': 210.351,
	'h': -38.3094,
	'i': 2.96344,
	'data_range': (4, 300),
	'equation_range': (4, 300),
	'units': 'J/(kg·K)',
	'fit_error': 5  # % error relative to data
}


def calculate_aluminum5083_property(coefficients, temperature):
	"""
	Calculate Aluminum 5083 material property using polynomial equation.

	Parameters:
	-----------
	coefficients : dict
		Dictionary containing coefficients a through i and metadata
	temperature : float or array-like
		Temperature in Kelvin

	Returns:
	--------
	float or array
		Property value in appropriate units

	Equation:
	y = 10^(a + b*log10(T) + c*(log10(T))^2 + d*(log10(T))^3 + e*(log10(T))^4 +
			f*(log10(T))^5 + g*(log10(T))^6 + h*(log10(T))^7 + i*(log10(T))^8)
	"""
	# Convert to numpy array for vectorized operations
	T = np.asarray(temperature)

	# Check temperature range
	min_temp, max_temp = coefficients['equation_range']
	if np.any(T < min_temp) or np.any(T > max_temp):
		print(f"Warning: Temperature outside recommended range {min_temp}-{max_temp} K")

	# Calculate log10(T)
	log_T = np.log10(T)

	# Calculate polynomial in log space
	exponent = (coefficients['a'] +
				coefficients['b'] * log_T +
				coefficients['c'] * log_T**2 +
				coefficients['d'] * log_T**3 +
				coefficients['e'] * log_T**4 +
				coefficients['f'] * log_T**5 +
				coefficients['g'] * log_T**6 +
				coefficients['h'] * log_T**7 +
				coefficients['i'] * log_T**8)

	# Return 10^exponent
	return 10**exponent


def thermal_conductivity(temperature):
	"""
	Calculate thermal conductivity for Aluminum 5083.

	Parameters:
	-----------
	temperature : float or array-like
		Temperature in Kelvin

	Returns:
	--------
	float or array
		Thermal conductivity in W/(m·K)
	"""
	return calculate_aluminum5083_property(THERMAL_CONDUCTIVITY_COEFFS, temperature)


def specific_heat(temperature):
	"""
	Calculate specific heat for Aluminum 5083.

	Parameters:
	-----------
	temperature : float or array-like
		Temperature in Kelvin

	Returns:
	--------
	float or array
		Specific heat in J/(kg·K)
	"""
	return calculate_aluminum5083_property(SPECIFIC_HEAT_COEFFS, temperature)


def thermal_diffusivity(temperature, density=2650):
	"""
	Calculate thermal diffusivity for Aluminum 5083.

	Parameters:
	-----------
	temperature : float or array-like
		Temperature in Kelvin
	density : float, optional
		Density in kg/m³ (default: 2650 kg/m³ for Al 5083)

	Returns:
	--------
	float or array
		Thermal diffusivity in m²/s
	"""
	k = thermal_conductivity(temperature)
	cp = specific_heat(temperature)
	return k / (density * cp)


def plot_aluminum5083_properties(temperature_range=(4, 300), num_points=1000):
	"""
	Plot all Aluminum 5083 material properties over specified temperature range.

	Parameters:
	-----------
	temperature_range : tuple
		(min_temp, max_temp) in Kelvin
	num_points : int
		Number of points for smooth curves
	"""
	# Create temperature array
	T = np.linspace(temperature_range[0], temperature_range[1], num_points)

	# Calculate properties
	k = thermal_conductivity(T)
	cp = specific_heat(T)
	alpha = thermal_diffusivity(T)

	# Create plots
	fig, axes = plt.subplots(2, 2, figsize=(12, 10))

	# Thermal conductivity
	axes[0, 0].plot(T, k, 'b-', linewidth=2)
	axes[0, 0].set_xlabel('Temperature (K)')
	axes[0, 0].set_ylabel('Thermal Conductivity (W/m·K)')
	axes[0, 0].set_title('Aluminum 5083 Thermal Conductivity')
	axes[0, 0].grid(True, alpha=0.3)

	# Thermal conductivity (log scale)
	axes[0, 1].loglog(T, k, 'b-', linewidth=2)
	axes[0, 1].set_xlabel('Temperature (K)')
	axes[0, 1].set_ylabel('Thermal Conductivity (W/m·K)')
	axes[0, 1].set_title('Aluminum 5083 Thermal Conductivity - Log Scale')
	axes[0, 1].grid(True, alpha=0.3)

	# Specific heat
	axes[1, 0].plot(T, cp, 'r-', linewidth=2)
	axes[1, 0].set_xlabel('Temperature (K)')
	axes[1, 0].set_ylabel('Specific Heat (J/kg·K)')
	axes[1, 0].set_title('Aluminum 5083 Specific Heat')
	axes[1, 0].grid(True, alpha=0.3)

	# Thermal diffusivity
	axes[1, 1].plot(T, alpha * 1e6, 'g-', linewidth=2)  # Convert to mm²/s for better scale
	axes[1, 1].set_xlabel('Temperature (K)')
	axes[1, 1].set_ylabel('Thermal Diffusivity (mm²/s)')
	axes[1, 1].set_title('Aluminum 5083 Thermal Diffusivity')
	axes[1, 1].grid(True, alpha=0.3)

	plt.tight_layout()
	plt.show()

	return fig, axes


def compare_with_room_temperature_values(temperature=298.15):
	"""
	Compare calculated values with typical room temperature literature values.

	Parameters:
	-----------
	temperature : float, optional
		Temperature in Kelvin (default: 298.15 K = 25°C)
	"""
	k = thermal_conductivity(temperature)
	cp = specific_heat(temperature)
	alpha = thermal_diffusivity(temperature)

	print(f"Aluminum 5083 Properties at {temperature:.1f} K ({temperature-273.15:.1f}°C):")
	print("-" * 60)
	print(f"Thermal Conductivity:    {k:.1f} W/(m·K)")
	print(f"Specific Heat:           {cp:.0f} J/(kg·K)")
	print(f"Thermal Diffusivity:     {alpha*1e6:.1f} mm²/s")
	print()
	print("Typical literature values for Al 5083 at room temperature:")
	print("Thermal Conductivity:    ~117 W/(m·K)")
	print("Specific Heat:           ~900 J/(kg·K)")
	print("Thermal Diffusivity:     ~49 mm²/s")


def get_all_coefficients():
	"""
	Return all coefficient sets for easy access.

	Returns:
	--------
	dict
		Dictionary containing all coefficient sets
	"""
	return {
		'thermal_conductivity': THERMAL_CONDUCTIVITY_COEFFS,
		'specific_heat': SPECIFIC_HEAT_COEFFS
	}


def get_property_at_temperature(property_name, temperature):
	"""
	Get a specific property at a given temperature.

	Parameters:
	-----------
	property_name : str
		'thermal_conductivity', 'specific_heat', or 'thermal_diffusivity'
	temperature : float or array-like
		Temperature in Kelvin

	Returns:
	--------
	float or array
		Property value
	"""
	if property_name == 'thermal_conductivity':
		return thermal_conductivity(temperature)
	elif property_name == 'specific_heat':
		return specific_heat(temperature)
	elif property_name == 'thermal_diffusivity':
		return thermal_diffusivity(temperature)
	else:
		raise ValueError(f"Unknown property: {property_name}")


if __name__ == "__main__":
	# Example usage and demonstration
	print("Aluminum 5083 Material Properties Calculator")
	print("=" * 45)

	# Test at a few key temperatures
	test_temps = [4, 20, 77, 150, 200, 273.15, 298.15, 300]  # Kelvin

	print(f"{'Temperature (K)':<15} {'Temp (°C)':<10} {'k (W/m·K)':<12} {'Cp (J/kg·K)':<12} {'α (mm²/s)':<12}")
	print("-" * 75)

	for T in test_temps:
		k = thermal_conductivity(T)
		cp = specific_heat(T)
		alpha = thermal_diffusivity(T) * 1e6  # Convert to mm²/s
		temp_c = T - 273.15
		print(f"{T:<15.1f} {temp_c:<10.1f} {k:<12.2f} {cp:<12.0f} {alpha:<12.2f}")

	print("\nComparing with literature values:")
	compare_with_room_temperature_values()

	print("\nGenerating plots...")
	plot_aluminum5083_properties()