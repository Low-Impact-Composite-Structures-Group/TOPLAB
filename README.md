# hydrogen_fuel_tank
This package enables the analysis of hydrogen fuel tank, providing insight into the thermo-mechanical loading during filling and draining of the tank.

## Running Analyses
Analysis and example files are run using python through the command line as follows:

~~~
python main.py path.to.main path/to/main.YAML
~~~

Where the first argument is a Python import path (module), and the second is the path to the YAML config file for the analysis.

Examples:

~~~
python main.py analysis.compare_dynamic_models examples/compare_dynamic_models/main.YAML
python main.py examples.compare_dynamic_models.main examples/compare_dynamic_models/main.YAML
~~~

Shorthand is also supported for common cases:

~~~
python main.py compare_dynamic_models examples/compare_dynamic_models/main.YAML
~~~

Each main file (with the exception for the Lin energy derivative) has at least a perform_analysis function which runs, with config input. Examples only have this function, where analysis files also have a save_data and plot_data method. This ensures that data is stored for reproducibility.

The deprecated files are run through the main_deprecated.py file. Here the desired module and associated perform_analysis function can be imported an run by calling perform_analysis().
