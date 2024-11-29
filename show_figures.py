import pickle
import sys
import matplotlib.pyplot as plt
from matplotlib import use as mpl_use

# Set the interactive backend
mpl_use('TkAgg')

def show_figures(file_paths):
    for file_path in file_paths:
        try:
            print(f"Loading figure from {file_path}...")
            with open(file_path, 'rb') as file:
                fig = pickle.load(file)
                if isinstance(fig, plt.Figure):
                    print(f"Successfully loaded figure from {file_path}. Displaying...")
                    plt.figure(fig.number)  # Ensure the figure is active
                else:
                    print(f"The loaded object from {file_path} is not a matplotlib figure.")
        except Exception as e:
            print(f"Failed to load figure from {file_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python show_figures.py <path_to_pkl_file1> <path_to_pkl_file2> ...")
    else:
        show_figures(sys.argv[1:])
        print("Displaying all figures. Close the figures to exit.")
        plt.show()  # Show all figures simultaneously