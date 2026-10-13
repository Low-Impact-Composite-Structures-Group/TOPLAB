import json
import numpy as np
import matplotlib.pyplot as plt


# Load JSON file
with open("rohacell.json", "r") as f:
    data = json.load(f)

# Extract data
temperature = np.array(data["temperature"], dtype=float)
thermal_conductivity = np.array(data["thermal_conductivity"], dtype=float)

# Combine into N x 2 array: [temperature, thermal conductivity]
properties = np.column_stack((temperature, thermal_conductivity))

#create second array with halved thermal conductivity values
halved_thermal_conductivity = thermal_conductivity / 2.0

quartered_thermal_conductivity = thermal_conductivity / 4.0

by_eight = thermal_conductivity / 8.0

data_modified = {
    "temperature": temperature.tolist(),
    "thermal_conductivity": halved_thermal_conductivity.tolist()
}

data_modified_quartered = {
    "temperature": temperature.tolist(),
    "thermal_conductivity": quartered_thermal_conductivity.tolist()
}

data_modified_by_eight = {
    "temperature": temperature.tolist(),
    "thermal_conductivity": by_eight.tolist()
}

with open("magic_foam_by_eight.json", "w") as f:
    json.dump(data_modified_by_eight, f, indent=4)

with open("magic_foam_quartered.json", "w") as f:
    json.dump(data_modified_quartered, f, indent=4)

with open("magic_foam.json", "w") as f:
    json.dump(data_modified, f, indent=4)

# plot thermal conductivity vs temperature
plt.figure(figsize=(8, 6))
plt.plot(temperature, thermal_conductivity, linestyle='-', color='b')
plt.plot(temperature, halved_thermal_conductivity, linestyle='-', color='r')
plt.plot(temperature, quartered_thermal_conductivity, linestyle='-', color='g')
plt.plot(temperature, by_eight, linestyle='-', color='m')
plt.title("Thermal Conductivity vs Temperature for Rohacell")
plt.xlabel("Temperature [K]")
plt.ylabel("Thermal Conductivity [W/m·K]")
plt.legend(["Original", "Halved", "Quartered", "By Eight"])
plt.grid(True)
plt.show()



print(properties)