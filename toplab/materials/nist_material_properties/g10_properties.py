import numpy as np
import matplotlib.pyplot as plt


# G10 material property coefficients from NIST data
# Equation: y = 10^(a + b*log10(T) + c*(log10(T))^2 + d*(log10(T))^3 + e*(log10(T))^4 + f*(log10(T))^5 + g*(log10(T))^6 + h*(log10(T))^7 + i*(log10(T))^8)
# Where T is temperature in K, and y is the property value

# Thermal Conductivity (Normal Direction) - W/(m·K)
THERMAL_CONDUCTIVITY_NORMAL_COEFFS = {
	'a': -4.1236,
	'b': 13.788,
	'c': -26.068,
	'd': 26.272,
	'e': -14.663,
	'f': 4.4954,
	'g': -0.6905,
	'h': 0.0397,
	'i': 0,
	'data_range': (4, 300),
	'equation_range': (10, 300),
	'units': 'W/(m·K)'
}

# Thermal Conductivity (Warp Direction) - W/(m·K)
THERMAL_CONDUCTIVITY_WARP_COEFFS = {
	'a': -2.64827,
	'b': 8.80228,
	'c': -24.8998,
	'd': 41.1625,
	'e': -39.8754,
	'f': 23.1778,
	'g': -7.95635,
	'h': 1.48806,
	'i': -0.11701,
	'data_range': (4, 300),
	'equation_range': (12, 300),
	'units': 'W/(m·K)'
}

# Specific Heat - J/(kg·K)
SPECIFIC_HEAT_COEFFS = {
	'a': -2.4083,
	'b': 7.6006,
	'c': -8.2982,
	'd': 7.3301,
	'e': -4.2386,
	'f': 1.4294,
	'g': -0.24396,
	'h': 0.015236,
	'i': 0,
	'data_range': (4, 300),
	'equation_range': (4, 300),
	'units': 'J/(kg·K)'
}


def calculate_g10_property(coefficients, temperature):
	"""
	Calculate G10 material property using polynomial equation.

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


def thermal_conductivity_normal(temperature):
	"""
	Calculate thermal conductivity in normal direction for G10.

	Parameters:
	-----------
	temperature : float or array-like
		Temperature in Kelvin

	Returns:
	--------
	float or array
		Thermal conductivity in W/(m·K)
	"""
	return calculate_g10_property(THERMAL_CONDUCTIVITY_NORMAL_COEFFS, temperature)


def thermal_conductivity_warp(temperature):
	"""
	Calculate thermal conductivity in warp direction for G10.

	Parameters:
	-----------
	temperature : float or array-like
		Temperature in Kelvin

	Returns:
	--------
	float or array
		Thermal conductivity in W/(m·K)
	"""
	return calculate_g10_property(THERMAL_CONDUCTIVITY_WARP_COEFFS, temperature)


def specific_heat(temperature):
	"""
	Calculate specific heat for G10.

	Parameters:
	-----------
	temperature : float or array-like
		Temperature in Kelvin

	Returns:
	--------
	float or array
		Specific heat in J/(kg·K)
	"""
	return calculate_g10_property(SPECIFIC_HEAT_COEFFS, temperature)


def plot_g10_properties(temperature_range=(4, 300), num_points=1000):
	"""
	Plot all G10 material properties over specified temperature range.

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
	k_normal = thermal_conductivity_normal(T)
	k_warp = thermal_conductivity_warp(T)
	cp = specific_heat(T)

	# Create plots
	fig, axes = plt.subplots(2, 2, figsize=(12, 10))

	# Thermal conductivity comparison
	axes[0, 0].loglog(T, k_normal, 'b-', label='Normal Direction', linewidth=2)
	axes[0, 0].loglog(T, k_warp, 'r-', label='Warp Direction', linewidth=2)
	axes[0, 0].set_xlabel('Temperature (K)')
	axes[0, 0].set_ylabel('Thermal Conductivity (W/m·K)')
	axes[0, 0].set_title('G10 Thermal Conductivity')
	axes[0, 0].legend()
	axes[0, 0].grid(True, alpha=0.3)

	# Thermal conductivity normal (linear scale)
	axes[0, 1].plot(T, k_normal, 'b-', linewidth=2)
	axes[0, 1].set_xlabel('Temperature (K)')
	axes[0, 1].set_ylabel('Thermal Conductivity (W/m·K)')
	axes[0, 1].set_title('G10 Thermal Conductivity (Normal) - Linear Scale')
	axes[0, 1].grid(True, alpha=0.3)

	# Specific heat
	axes[1, 0].plot(T, cp, 'g-', linewidth=2)
	axes[1, 0].set_xlabel('Temperature (K)')
	axes[1, 0].set_ylabel('Specific Heat (J/kg·K)')
	axes[1, 0].set_title('G10 Specific Heat')
	axes[1, 0].grid(True, alpha=0.3)

	# Specific heat (log scale)
	axes[1, 1].semilogx(T, cp, 'g-', linewidth=2)
	axes[1, 1].set_xlabel('Temperature (K)')
	axes[1, 1].set_ylabel('Specific Heat (J/kg·K)')
	axes[1, 1].set_title('G10 Specific Heat - Log Scale')
	axes[1, 1].grid(True, alpha=0.3)

	plt.tight_layout()
	plt.show()

	return fig, axes