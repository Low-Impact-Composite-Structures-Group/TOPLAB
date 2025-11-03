import os
import yaml
import pickle

from typing import Protocol

from importlib import import_module

class Figure(Protocol):
    def show(self):
        ...


class AnalysisProtocol(Protocol):

    def perform_analysis(self, config: dict) -> dict:
        ...

    def extract_data(self, data: dict, store_path: str) -> None:
        ...

    def plot_results(self, store_path: str, fig_path: str, config: dict, extensions: list[str]) -> Figure:
        ...


def run_analysis(module_name: str, config_file: str, data_dir: str = "data"):
    # Load the analysis module dynamically
    analysis: AnalysisProtocol = import_module(module_name)

    # Load YAML config
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    # Construct file paths
    base_name = os.path.splitext(os.path.basename(config_file))[0]
    dir_name = os.path.basename(os.path.dirname(config_file))
    data_dir = os.path.join(data_dir, dir_name)

    os.makedirs(data_dir, exist_ok=True)

    pickle_path = os.path.join(data_dir, f"{base_name}.pkl")
    store_path = os.path.join(data_dir, f"{base_name}.npz")
    fig_path = os.path.join(data_dir, f"{base_name}")

    # Reuse or recompute
    if os.path.exists(pickle_path):
        user_input = input("Pickle exists. Recompute? (y/n): ").lower()
        if user_input in ["y", "yes"]:
            print("Performing analysis...")
            data = analysis.perform_analysis(config)
            print("Analysis done!")
            data["config"] = config
        else:
            print("Opening pickle...")
            with open(pickle_path, "rb") as f:
                data = pickle.load(f)
            print("Look @ me I'm a pickle! Pickle loaded!")
    else:
        data = analysis.perform_analysis(config)

    # Add config to data, for completeness
    data["config"] = config

    user_input = input("Save to pickle? (y/n): ").lower()
    if user_input in ["y", "yes"]:
        print("Saving pickle...")
        with open(pickle_path, "wb") as f:
            pickle.dump(data, f)
        print("Pickle saved!")

    # Extract and plot
    print("Extracting data...")
    analysis.extract_data(data, store_path)
    print("Data extracted!")
    print("Plotting...")
    fig: Figure = analysis.plot_results(store_path, fig_path, config.get("plotting"))
    print("Plotting completed!")
    if fig is not None:
        fig.show()


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python main.py <analysis_module> <config_file>")
        print("Example: python main.py switch_drain_analysis config/switch_drain.yaml")
    else:
        run_analysis(sys.argv[1], sys.argv[2])
