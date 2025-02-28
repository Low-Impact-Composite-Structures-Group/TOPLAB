import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
file_path = 'NOVAC00_V01_SIMULATION_RESULTS.CSV'
data = pd.read_csv(file_path)

# Function to truncate data
def truncate_data(data, percentage):
    if percentage <= 0 or percentage > 100:
        raise ValueError("Percentage must be between 0 and 100")
    step = int(100 / percentage)
    return data[::step]

# Specify the percentage of data to keep
percentage_to_keep = 1  # Change this value as needed

# Truncate the data
data = truncate_data(data, percentage_to_keep)

# Extract the time column
time = data['time [s]'].values

# Plot each column against time
for column in data.columns[1:]:
    plt.figure()
    plt.plot(time, data[column].values)
    plt.title(f'{column} vs Time')
    plt.xlabel('Time [s]')
    plt.ylabel(column)
    plt.grid(True)
    plt.xticks()
    plt.show()