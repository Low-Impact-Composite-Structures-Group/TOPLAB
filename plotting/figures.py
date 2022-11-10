
from dataclasses import dataclass

import matplotlib.pyplot as plt


@dataclass
class Line:
    x_data: list[float]
    y_data: list[float]
    label: str
    marker: None = ""   # Set to None to create with marker


@dataclass
class TwinXFigure:
    data: list[list[Line]]
    x_label: str
    y_labels: list[str]
    x_ticks: list[float] = None
    y_ticks: list[list[float]] = None

    def __post_init__(self):
        self.create_figure_and_axis()
        self.plot_data()
        self.format_ticks()
        self.format_labels()
        self.format_legend()


    def create_figure_and_axis(self):
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(111)
        self.ax2 = self.ax1.twinx()
        self.ax2._get_lines.prop_cycler = self.ax1._get_lines.prop_cycler
        self.ax = [self.ax1, self.ax2]
        return self.fig, self.ax

    def plot_data(self):
        self.plots = [
            ax.plot(
                line.x_data,
                line.y_data,
                label=line.label,
                marker=line.marker
            )[0]
            for ax, ax_data in zip(self.ax, self.data)
            for line in ax_data
        ]

    def format_labels(self):
        self.ax1.set_xlabel(self.x_label)
        for ax, label in zip(self.ax, self.y_labels):
            ax.set_ylabel(label)
        self.fig.tight_layout()

    def format_legend(self):
        labs = [l.get_label() for l in self.plots]
        self.ax1.legend(self.plots, labs, loc=0)

    def format_ticks(self):
        if self.x_ticks is not None:
            self.ax1.set_xlim((self.x_ticks[0], self.x_ticks[-1]))
            self.ax1.set_xticks(self.x_ticks)
        if self.y_ticks is not None:
            for ax, y_ticks in zip(self.ax, self.y_ticks):
                if y_ticks is not None:
                    ax.set_ylim((y_ticks[0], y_ticks[-1]))
                    ax.set_yticks(y_ticks)
        self.ax1.grid()

    def show(self):
        self.fig.show()


def main():
    pass


if __name__ == "__main__":
    main()


# End
