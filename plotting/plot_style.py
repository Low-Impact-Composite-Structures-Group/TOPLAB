"""Plot style is used to define the default style. This is achieved by
setting the plot cyclers.

Fuel Tank - Plot Style
Hydrogen Storage in Civil Aviation PhD
Victor Kees Poorte, 2022
"""

from __future__ import annotations
from dataclasses import dataclass
import math
import matplotlib.pyplot as plt

from cycler import cycler


FONT_SIZE = 11          # [pt]
FIGURE_WIDTH = 10.0     # [cm]
FIGURE_HEIGHT = 8.8     # [cm]
DEFAULT_FONT = "Times New Roman"


CM2INCH = 0.393701


class MyCycler(object):

    def __init__(self) -> None:
        self.colors = [
            "#00A6D6",  # Delft Blue
            "#E03C31",  # Dark orange
            "#009B77",  # Dark Green
            "#A50034",  # Dark Red
            "#6F1D77",   # Purple
        ]
        self.linestyles = [
            "-",
            "--",
            "-.",
            ":",
            (0, (3, 10, 1, 10, 1, 10))
        ]
        self.markers = [
            "o", "s", "D", "v", "p"
        ]

    def get_cycler(self):
        """Method to get the personalized cycler"""
        return (
            cycler(color=self.colors)
            + cycler(linestyle=self.linestyles)
            + cycler(marker=self.markers)
        )


def set_font(font_name: str = DEFAULT_FONT):
    """Set the font family for all matplotlib plots.

    Args:
        font_name: The name of the font family to use (e.g., "Cambria", "Arial", etc.)
    """
    plt.rcParams.update({'font.family': font_name})


# Set default font and other parameters
set_font()
plt.rcParams.update({'font.size': FONT_SIZE})
plt.rcParams["figure.figsize"] = (
    FIGURE_WIDTH * CM2INCH, FIGURE_HEIGHT * CM2INCH
)
plt.rc("axes", prop_cycle=MyCycler().get_cycler())


@dataclass
class MyFigure:
    x_data: list[list[float]]
    y_data: list[list[float]]
    labels: list[str]
    xlabel: str
    ylabel: str
    filename: str = None

    def __post_init__(self):
        self.create_figure()
        self.set_axis_labels()
        self.plot_data()
        self.create_legend()
        # self.create_x_ticks()
        # self.create_y_ticks()
        self.ax.grid()

    def create_figure(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

    def set_axis_labels(self):
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)

    def plot_data(self):
        self.lines = list()
        for x, y, label in zip(self.x_data, self.y_data, self.labels):
            self.lines += self.ax.plot(
                x, y, label=label, marker=""
            )
        return self.lines

    def create_legend(self):
        labs = [l.get_label() for l in self.lines]
        self.ax.legend(self.lines, labs)

    def create_x_ticks(self):

        xticks = TickFormatter(self.x_data[0]).ticks
        self.ax.set_xticks(xticks)
        self.ax.set_xlim((xticks[0], xticks[-1]))

    def create_y_ticks(self):

        xticks = TickFormatter(self.y_data[0]).ticks
        self.ax.set_yticks(xticks)
        self.ax.set_ylim((xticks[0], xticks[-1]))


@dataclass
class AxisValue:
    value: float

    @property
    def sign(self) -> int:
        if self.value >= 0:
            return +1
        return -1

    @property
    def order_of_magnitude(self) -> int:
        if self.value == 0:
            return 0
        # This line has been added to account for floating points
        log_value = math.log(abs(self.value), 10)
        if log_value % 1 > 0.99:
            return round(log_value)
        return math.floor(log_value)

    @property
    def limit_value(self) -> float:
        ...


class UpperValue(AxisValue):


    @property
    def limit_value(self) -> float:
        return math.ceil(self.value / 10 ** self.order_of_magnitude) * 10 ** self.order_of_magnitude


class LowerValue(AxisValue):


    @property
    def limit_value(self) -> float:
        return math.floor(self.value / 10 ** self.order_of_magnitude) * 10 ** self.order_of_magnitude


class StepValue(AxisValue):

    @property
    def indicative_step(self):
        value = math.ceil(self.value / 10 ** self.order_of_magnitude) * 10 ** self.order_of_magnitude
        if value % 2 < 1:
            return value
        return value - 1


@dataclass
class TickFormatter:
    data: list[float]
    steps: int = 6
    max_steps: int = 10

    @property
    def lower(self):
        return LowerValue(self.data[0])

    @property
    def upper(self):
        return UpperValue(self.data[-1])

    @property
    def delta(self):
        return self.upper.value - self.lower.value

    @property
    def step(self):
        if self.upper.order_of_magnitude <= 1:
            return StepValue(self.delta / self.steps).indicative_step
        if self.upper.order_of_magnitude == 2:
            return 50
        return 10 ** (self.upper.order_of_magnitude - 1)

    @property
    def ticks(self):
        ticks = [self.lower.value]
        for _ in range(self.max_steps):
            ticks.append(ticks[-1] + self.step)
            if ticks[-1] > self.upper.value:
                return ticks






def define_axis_ticks(axis_values: list[float]) -> list[float]:
    pass

def define_number_order(number: float) -> int:
    pass


def main():
    pass


if __name__ == "__main__":
    main()


# End