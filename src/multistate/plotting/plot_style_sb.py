"""
Seaborn-based plot styling for hydrogen fuel tank visualization.
This provides a modernized styling approach, colocated under src/multistate/plotting.

Hydrogen Storage in Civil Aviation PhD
"""

from typing import List, Optional, Tuple
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.font_manager as fm

# Constants that might be useful
CM_TO_INCH = 0.393701

# Global font settings
FONT_SIZE = 16
FONT_NAME = "DejaVu Sans"
LEGEND_FONT_SIZE = 14

DONKERBLAUW = "#0C2340"  # Dark Blue
TURKOOIS = "#00B8C8"  # Turquoise
KONINGSBLAUW = "#0076C2"  # Royal Blue
PAARS = "#6F1D77" # Purple
ROZE = "#EF60A3"  # Pink
BORDEAUX = "#A50034"  # Bordeaux Red
ROOD = "#E03C31"  # Red
ORANJE = "#EC6842" # Orange
GEEL = "#FFB81C"  # Yellow
GROEN = "#6CC24A"  # Green
BOSGROEN = "#009B77"  # Forest Green
DONKERGRIJS = "#4B4B4D"  # Dark Gray



# Delft University color palette
DELFT_PALETTE = [DONKERBLAUW, TURKOOIS, KONINGSBLAUW, PAARS, ROZE, BORDEAUX, ROOD, ORANJE, GEEL, BOSGROEN, DONKERGRIJS]

# Palette mapping for different styles
PALETTE_MAP = {
	"delft": DELFT_PALETTE,
	"Set2": "Set2",
	"viridis": "viridis",
	"plasma": "plasma"
}

# Legend positioning options
LEGEND_BBOX_TO_ANCHOR = {
	"upper_right": (1.0, 1.0),
	"upper_left": (0.0, 1.0),
	"lower_right": (1.0, 0.0),
	"lower_left": (0.0, 0.0),
	"center": (0.5, 0.5),
	"outside_right": (1.02, 1.0)
}

# Global variables for current settings
CURRENT_PALETTE = DELFT_PALETTE
LEGEND_BBOX_TO_ANCHOR_DEFAULT = (1.02, 1.0)

# Define the path to our custom fonts directory (local to this module)
FONTS_DIR = os.path.join(os.path.dirname(__file__), 'fonts')

def register_custom_fonts():
	"""Register custom fonts from the local fonts directory."""
	if not os.path.exists(FONTS_DIR):
		return False

	# Count how many fonts we register
	font_count = 0

	# Register all font files in the directory
	for font_file in os.listdir(FONTS_DIR):
		if font_file.endswith(('.ttf', '.otf')):
			font_path = os.path.join(FONTS_DIR, font_file)
			if os.path.getsize(font_path) > 0:  # Skip empty placeholder files
				fm.fontManager.addfont(font_path)
				font_count += 1
			else:
				continue

	# Force matplotlib to rebuild the font cache
	try:
		fm.fontManager._load_fontmanager()
	except Exception:
		try:
			fm._get_fontconfig_fonts.cache_clear()
		except Exception:
			pass
		fm._load_fontmanager(try_read_cache=False)

	return font_count > 0

def show_available_fonts():
	"""Print a list of all available fonts that can be used with matplotlib."""
	fonts = sorted(set([f.name for f in fm.fontManager.ttflist]))
	print("Available fonts:")
	for font in fonts:
		print(f"  - {font}")
	return fonts

def set_font_with_fallbacks(primary_font=FONT_NAME, fallbacks=["Times New Roman", "Arial", "DejaVu Serif"]):
	"""Attempt to set the primary font, falling back to alternatives if not available."""
	available_fonts = set([f.name for f in fm.fontManager.ttflist])

	if primary_font in available_fonts:
		plt.rcParams["font.family"] = primary_font
		return primary_font

	for font in fallbacks:
		if font in available_fonts:
			plt.rcParams["font.family"] = font
			return font

	return plt.rcParams["font.family"]


def set_seaborn_style(font: str = "Cambria", palette: str = "delft",
					  style: str = "whitegrid", context: str = "paper"):
	"""Set the global style for Seaborn plots with custom font support."""
	ensure_fonts_registered()

	if palette == "delft":
		palette = DELFT_PALETTE

	sns.set_theme(style=style, palette=palette, context=context)

	set_font_with_fallbacks(font)


def update_font_settings(master_size=None, legend_size=None, font_name=None):
	"""Update global font settings dynamically."""
	global FONT_SIZE, LEGEND_FONT_SIZE, FONT_NAME

	if master_size is not None:
		FONT_SIZE = master_size
		plt.rcParams['font.size'] = FONT_SIZE
	if legend_size is not None:
		LEGEND_FONT_SIZE = legend_size
		plt.rcParams['legend.fontsize'] = LEGEND_FONT_SIZE
	if font_name is not None:
		FONT_NAME = font_name
		plt.rcParams['font.family'] = FONT_NAME


def configure_plot_style(font="Cambria", palette="delft", bbox_to_anchor_key=None, legend_position=None,
						style="white", context="paper", figure_size=None, dpi=None, **kwargs):
	"""Configure the global plot style settings."""
	global CURRENT_PALETTE, LEGEND_BBOX_TO_ANCHOR_DEFAULT

	if font == "Cambria":
		font = FONT_NAME

	plt.rcParams['font.family'] = font
	plt.rcParams['font.size'] = FONT_SIZE
	plt.rcParams['legend.fontsize'] = LEGEND_FONT_SIZE

	sns.set_theme(style="white", palette=PALETTE_MAP.get(palette, "Set2"), rc={
		'font.family': font,
		'font.size': FONT_SIZE,
		'axes.titlesize': FONT_SIZE,
		'axes.labelsize': FONT_SIZE,
		'xtick.labelsize': FONT_SIZE,
		'ytick.labelsize': FONT_SIZE,
		'legend.fontsize': LEGEND_FONT_SIZE,
		'figure.titlesize': FONT_SIZE
	})

	CURRENT_PALETTE = PALETTE_MAP.get(palette, "Set2")

	if bbox_to_anchor_key and bbox_to_anchor_key in LEGEND_BBOX_TO_ANCHOR:
		LEGEND_BBOX_TO_ANCHOR_DEFAULT = LEGEND_BBOX_TO_ANCHOR[bbox_to_anchor_key]
	elif legend_position:
		LEGEND_BBOX_TO_ANCHOR_DEFAULT = legend_position

def get_palette_colors(n_colors: int = 10, palette: str = None):
	"""Get a list of colors from the specified palette or current palette."""
	return sns.color_palette(palette, n_colors=n_colors)


def plot_line(x_data: List[float], y_data: List[float], label: str,
			  ax=None, color=None, linestyle='-', marker=None, alpha=1.0):
	"""Plot a line using Seaborn."""
	if ax is None:
		ax = plt.gca()
	sns.lineplot(x=x_data, y=y_data, label=label, ax=ax,
				 color=color, linestyle=linestyle, marker=marker, alpha=alpha)


def plot_scatter(x_data: List[float], y_data: List[float], label: str,
				ax=None, color=None, marker='o', size=50, alpha=0.7):
	"""Plot a scatter plot using Seaborn."""
	if ax is None:
		ax = plt.gca()
	sns.scatterplot(x=x_data, y=y_data, label=label, ax=ax,
				   color=marker and color, marker=marker, s=size, alpha=alpha)


def create_figure_with_ax(figsize: Tuple[float, float] = (8, 6)) -> Tuple:
	"""Create a figure and axis with the given size."""
	fig, ax = plt.subplots(figsize=figsize)
	return fig, ax


def apply_custom_ticks(ax, xticks=None, yticks=None,
					   xmin=None, xmax=None, ymin=None, ymax=None):
	"""Apply custom ticks and limits to an axis."""
	if xticks is not None:
		ax.set_xticks(xticks)
	if yticks is not None:
		ax.set_yticks(yticks)

	if xmin is not None and xmax is not None:
		ax.set_xlim(xmin, xmax)
	if ymin is not None and ymax is not None:
		ax.set_ylim(ymin, ymax)


def format_axis_labels(ax, xlabel=None, ylabel=None, title=None):
	"""Format axis labels."""
	if xlabel:
		ax.set_xlabel(xlabel)
	if ylabel:
		ax.set_ylabel(ylabel)
	if title:
		ax.set_title(title)


def add_legend(ax, loc='best', frameon=True, title=None):
	"""Add a legend to the plot."""
	ax.legend(loc=loc, frameon=frameon, title=title)


def use_delft_palette():
	"""Use the Delft University color palette."""
	set_seaborn_style(palette="delft")
	return DELFT_PALETTE


# Only register fonts if this module is imported directly
_registered = False
def ensure_fonts_registered():
	global _registered
	if not _registered:
		register_custom_fonts()
		_registered = True

ensure_fonts_registered()


if __name__ == "__main__":
	configure_plot_style(font="Cambria", palette="delft")
	fig, ax = create_figure_with_ax()
	x = np.linspace(0, 10, 100)
	plot_line(x, np.sin(x), "Sine Wave", ax=ax)
	plot_line(x, np.cos(x), "Cosine Wave", ax=ax)
	format_axis_labels(ax, xlabel="X Value", ylabel="Y Value", title="Example Plot")
	add_legend(ax)