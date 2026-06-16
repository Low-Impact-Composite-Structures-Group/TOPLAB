import os
import yaml
import pickle
import inspect
import sys
import warnings

from typing import Protocol

from importlib import import_module
import importlib.util


def _import_module_from_path(path: str):
    module_id = f"analysis_from_path_{abs(hash(os.path.abspath(path)))}"
    spec = importlib.util.spec_from_file_location(module_id, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from path: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_analysis_module(module_name: str):
    """Resolve user-provided module names.

    Supports:
    - Regular import paths (e.g. analysis.compare_dynamic_models)
    - Shorthands (e.g. compare_dynamic_models -> analysis.compare_dynamic_models)
    - File/dir paths (e.g. examples/compare_dynamic_models/main.py or examples/compare_dynamic_models)
    """

    def _try_import(name: str):
        try:
            return import_module(name)
        except ModuleNotFoundError as exc:
            # Only treat as "module not found" if the missing name is *this* module.
            # If an internal dependency is missing, re-raise so the traceback is preserved.
            if getattr(exc, "name", None) == name:
                return None
            raise

    candidates: list[str] = [module_name]
    if "." not in module_name:
        candidates.extend(
            [
                f"examples.{module_name}.main",
                f"analysis.{module_name}",
                f"examples.{module_name}",
            ]
        )

    for candidate in candidates:
        mod = _try_import(candidate)
        if mod is not None:
            return mod

    # Fall back to path-based resolution
    path_candidates: list[str] = []
    user_path = os.path.expanduser(module_name)

    if user_path.endswith(".py"):
        path_candidates.append(user_path)
    else:
        path_candidates.extend(
            [
                user_path,
                os.path.join(user_path, "main.py"),
                os.path.join("examples", user_path, "main.py"),
                os.path.join("examples", user_path, "__init__.py"),
                os.path.join("analysis", f"{user_path}.py"),
            ]
        )

    for p in path_candidates:
        if os.path.isfile(p):
            return _import_module_from_path(p)

    attempted = "\n".join(f"  - {c}" for c in candidates)
    raise ModuleNotFoundError(
        "Could not import analysis module.\n"
        f"Requested: {module_name}\n"
        f"Tried import paths:\n{attempted}\n"
        f"Python: {sys.executable}"
    )

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
    analysis: AnalysisProtocol = _resolve_analysis_module(module_name)

    # Load YAML config
    try:
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Config file not found: {config_file}\n"
            "Tip: paths on some systems are case-sensitive (e.g. main.YAML vs main.yaml)."
        )

    # Construct file paths
    base_name = os.path.splitext(os.path.basename(config_file))[0]
    dir_name = os.path.basename(os.path.dirname(config_file))
    data_dir = os.path.join(data_dir, dir_name)

    os.makedirs(data_dir, exist_ok=True)

    pickle_path = os.path.join(data_dir, f"{base_name}.pkl")
    store_path = os.path.join(data_dir, f"{base_name}.npz")
    fig_path = os.path.join(data_dir, f"{base_name}")

    def _is_interactive() -> bool:
        # Pytest often runs with a TTY but should never block on prompts.
        if os.environ.get("PYTEST_CURRENT_TEST"):
            return False
        return sys.stdin.isatty()

    def _prompt_yes_no(prompt: str, default: bool = False) -> bool:
        if not _is_interactive():
            return default
        user_input = input(prompt).strip().lower()
        if user_input in {"y", "yes"}:
            return True
        if user_input in {"n", "no"}:
            return False
        return default

    # Reuse or recompute
    if os.path.exists(pickle_path):
        if _prompt_yes_no("Pickle exists. Recompute? (y/n): ", default=False):
            print("Performing analysis...")
            data = analysis.perform_analysis(config)
            print("Analysis done!")
            if data is None:
                data = {}
        else:
            print("Opening pickle...")
            with open(pickle_path, "rb") as f:
                data = pickle.load(f)
            print("Look @ me I'm a pickle! Pickle loaded!")
    else:
        data = analysis.perform_analysis(config)

    if data is None:
        data = {}

    # Add config to data, for completeness (only when the result is a dict)
    if isinstance(data, dict):
        data["config"] = config

    if _prompt_yes_no("Save to pickle? (y/n): ", default=False):
        print("Saving pickle...")
        with open(pickle_path, "wb") as f:
            pickle.dump(data, f)
        print("Pickle saved!")

    # Extract and plot (optional for lightweight example modules)
    extract_data = getattr(analysis, "extract_data", None)
    if callable(extract_data):
        print("Extracting data...")
        # Backward-compatible calling convention:
        # - Newer modules: extract_data(data: dict, store_path: str)
        # - Legacy modules: extract_data(pickle_path: str, store_path: str)
        sig = inspect.signature(extract_data)
        params = list(sig.parameters.values())

        if len(params) >= 2:
            first_param_name = params[0].name
            if first_param_name in {"pickle_path", "pkl_path", "picklefile"}:
                extract_data(pickle_path, store_path)
            else:
                extract_data(data, store_path)
        elif len(params) == 1:
            extract_data(data)
        else:
            extract_data()
        print("Data extracted!")

    plot_results = getattr(analysis, "plot_results", None)
    if callable(plot_results):
        print("Plotting...")
        fig: Figure = plot_results(store_path, fig_path, config.get("plotting"))
        print("Plotting completed!")
        if fig is not None and _is_interactive():
            fig.show()


if __name__ == "__main__":
    import sys
    warnings.warn(
        "main.py is a compatibility entrypoint. Prefer the driver + YAML workflow under analysis/multistate_systems/.",
        DeprecationWarning,
        stacklevel=1,
    )
    if len(sys.argv) != 3:
        print("Usage: python main.py <analysis_module> <config_file>")
        print("Compatibility example: python main.py analysis.compare_dynamic_models examples/compare_dynamic_models/main.yaml")
        print("Preferred workflow: python analysis/multistate_systems/DSE/driver_discharge.py")
    else:
        run_analysis(sys.argv[1], sys.argv[2])
